///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltru�aitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltru�aitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

//
// Copyright (c) 2016-2017 Vinnie Falco (vinnie dot falco at gmail dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Official repository: https://github.com/boostorg/beast
//

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <iostream>
#include <thread>

// dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/base64/base64_kernel_1.h>

#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>

// OpenCV includes
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = boost::asio::ip::tcp;

class Openface {
    public:
    LandmarkDetector::FaceModelParameters det_parameters;
    LandmarkDetector::CLNF face_model;
    FaceAnalysis::FaceAnalyser face_analyser;
    LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn;

    public:
    explicit Openface(
        LandmarkDetector::FaceModelParameters det_parameters,
        LandmarkDetector::CLNF face_model,
        FaceAnalysis::FaceAnalyser face_analyser,
        LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn)
          : det_parameters { det_parameters }
          , face_model { face_model }
          , face_analyser { face_analyser }
          , face_detector_mtcnn { face_detector_mtcnn } {
    }

    string run(vector<unsigned char> data) {
        ostringstream sout;

        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        cv::Mat_<uchar> grayscale_image;
        cv::cvtColor(img, grayscale_image, cv::COLOR_BGR2GRAY);

        // Detect faces in an image
        vector<cv::Rect_<float> > face_detections;
        
        vector<float> confidences;
        LandmarkDetector::DetectFacesMTCNN(face_detections, img, face_detector_mtcnn, confidences);
        // perform landmark detection for every face detected
        for (size_t face = 0; face < face_detections.size(); ++face)
        {
            // if there are multiple detections go through them
            bool success = LandmarkDetector::DetectLandmarksInImage(img, face_detections[face], face_model, det_parameters, grayscale_image);
            face_analyser.PredictStaticAUsAndComputeFeatures(img, face_model.detected_landmarks);

            sout << confidences[face] << " ";
            vector<pair<string, double>> intensities = face_analyser.GetCurrentAUsReg();
            for (vector<pair<string, double>>::iterator iter = intensities.begin(); iter != intensities.end(); ++iter) {
                sout << iter->second << " ";
            }
            sout << endl;
        }
        return sout.str();
    }
};

// Report a failure
void
fail(beast::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

// This function produces an HTTP response for the given
// request. The type of the response object depends on the
// contents of the request, so the interface requires the
// caller to pass a generic lambda for receiving the response.
template<
    class Body, class Allocator,
    class Send>
void
handle_request(
    http::request<Body, http::basic_fields<Allocator>>&& req,
    Send&& send,
    Openface openface)
{
    // Returns a not found response
    auto const not_found =
    [&req](beast::string_view target)
    {
        http::response<http::string_body> res{http::status::not_found, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.keep_alive(req.keep_alive());
        res.prepare_payload();
        return res;
    };
    if( req.method() != http::verb::post)
        return send(not_found(req.target()));

    http::response<http::string_body> res{http::status::ok, req.version()};
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.keep_alive(req.keep_alive());
    res.body() = openface.run(req.body());;
    res.prepare_payload();
    return send(std::move(res));
}

// This is the C++11 equivalent of a generic lambda.
// The function object is used to send an HTTP message.
template<class Stream>
struct send_lambda
{
    Stream& stream_;
    bool& close_;
    beast::error_code& ec_;

    explicit
    send_lambda(
        Stream& stream,
        bool& close,
        beast::error_code& ec)
        : stream_(stream)
        , close_(close)
        , ec_(ec)
    {
    }

    template<bool isRequest, class Body, class Fields>
    void
    operator()(http::message<isRequest, Body, Fields>&& msg) const
    {
        // Determine if we should close the connection after
        close_ = msg.need_eof();

        // We need the serializer here because the serializer requires
        // a non-const file_body, and the message oriented version of
        // http::write only works with const messages.
        http::serializer<isRequest, Body, Fields> sr{msg};
        http::write(stream_, sr, ec_);
    }
};

// Handles an HTTP server connection
void
do_session(
    tcp::socket& socket,
    Openface openface)
{
    bool close = false;
    beast::error_code ec;

    // This buffer is required to persist across reads
    beast::flat_buffer buffer;

    // This lambda is used to send messages
    send_lambda<tcp::socket> lambda{socket, close, ec};

    for(;;)
    {
        // Read a request
        // http::request<http::string_body> req;
        http::request<http::vector_body<unsigned char>> req;
        http::read(socket, buffer, req, ec);
        if(ec == http::error::end_of_stream)
            break;
        if(ec)
            return fail(ec, "read");

        // Send the response
        handle_request(std::move(req), lambda, openface);
        if(ec)
            return fail(ec, "write");
        if(close)
        {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            break;
        }
    }

    // Send a TCP shutdown
    socket.shutdown(tcp::socket::shutdown_send, ec);

    // At this point the connection is closed gracefully
}


int main(int argc, char* argv[])
{
    // Check command line arguments.
    if (argc != 3)
    {
        std::cerr <<
            "Usage: FaceLandmarkImgServer <address> <port> <doc_root>\n" <<
            "Example:\n" <<
            "    FaceLandmarkImgServer 0.0.0.0 8080\n";
        return EXIT_FAILURE;
    }
    vector<string> arguments{argv[0]};

    LandmarkDetector::FaceModelParameters det_parameters(arguments);

    // The modules that are being used for tracking
    cout << "Loading the model" << endl;
    LandmarkDetector::CLNF face_model(det_parameters.model_location);

    if (!face_model.loaded_successfully)
    {
        cout << "ERROR: Could not load the landmark detector" << endl;
        return 1;
    }

    cout << "Model loaded" << endl;

    // Load facial feature extractor and AU analyser (make sure it is static)
    FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
    face_analysis_params.OptimizeForImages();
    FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

    // If bounding boxes not provided, use a face detector
    cv::CascadeClassifier classifier(det_parameters.haar_face_detector_location);
    dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
    LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn(det_parameters.mtcnn_face_detector_location);

    if (!face_model.eye_model)
    {
        cout << "WARNING: no eye model found" << endl;
    }

    if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
    {
        cout << "WARNING: no Action Unit models found" << endl;
    }
    Openface openface(det_parameters, face_model, face_analyser, face_detector_mtcnn);

    try
    {
        auto const address = net::ip::make_address(argv[1]);
        auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
        // The io_context is required for all I/O
        net::io_context ioc{1};
        // The acceptor receives incoming connections
        tcp::acceptor acceptor{ioc, {address, port}};
        for(;;)
        {
            // This will receive the new connection
            tcp::socket socket{ioc};
            // Block until we get a connection
            acceptor.accept(socket);
            // Launch the session, transferring ownership of the socket
            std::thread{std::bind(
                &do_session,
                std::move(socket),
                openface)}.detach();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}