---
layout: post
title: Convolutions with cuDNN
date:   2021-01-02 13:29:52 -0600
categories: machine-learning GPGPU
---

This is just a brief tutorial in generating an edge detector with a conovlution
operation using cuDNN. For those who aren't familiar cuDNN is the deep learning 
primitive framework used as a backend for PyTorch and other higher-level 
mainstream deep learning frameworks. 


{% highlight cpp %}
#include <iostream>
#include <cudnn.h>
#include <cassert>
#include <cstdlib>

/****************************************************************

 FORWARD CONVOLUTION OPERATION 

****************************************************************/


// save an image to the working directory 
cv::Mat load_image(const char* image_path) {
	cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
	image.convertTo(image, CV_32FC3);
	cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
	return image;
}

void show_image(cv::Mat img) {
	cv::imshow("Test image", img); 
	int k = cv::waitKey(0); // Wait for a keystroke in the window 
}

void SAVE_IMAGE(const char* output_filename,
    float* c_buffer_ptr,
    int height,
    int width) 
{
    // create cv mat object from buffer and dimensions
    cv::Mat image(height, width, CV_32FC3, c_buffer_ptr);

    // convert negative values to 0 
    cv::threshold(image,
        image,
        /*threshold=*/0,
        /*maxval=*/0,
        cv::THRESH_TOZERO);

    // normalize the image pixels for greater contrast and visibility in output img
    cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
    
    // convert image to appropriate bit depth (8) and number of channels (3) 
    image.convertTo(image, CV_8UC3);

    // write to file
    cv::imwrite(output_filename, image);
    std::cerr << "Image output file directory: " << output_filename << std::endl;
}

// macro checkCUDAERRORS: return the error code and which line 
// it was found on from the cudnnStatus_t object (if errors found)
#define checkCUDAERRORS(expr)                                \
  {                                                          \
    cudnnStatus_t status = (expr);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "CUDNN ERROR ON LINE " << __LINE__ << ": "\
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }



int main(int argc, const char* argv[]) 
{
    // get GPU being used
    int GPU = (argc > 2) ? std::atoi(argv[2]) : 0;
    std::cerr << "GPU: " << GPU << std::endl;
    
    // sigmoid activation function implementation 
    bool sigmoid_usage = (argc > 3) ? std::atoi(argv[3]) : 0;
    std::cerr << "With sigmoid: " << std::boolalpha << sigmoid_usage << std::endl;

    // get image file and load into mat obj
    cv::Mat image = load_image("crack.jpg");

    cudaSetDevice(1);

    // get cuda context object 
    cudnnHandle_t CUDNN_CONTEXT;
    cudnnCreate(&CUDNN_CONTEXT);

    // instantiate tensor descriptors/context objects 
    cudnnTensorDescriptor_t input_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t convolution_desc;
    cudnnTensorDescriptor_t output_descriptor;

    // define tensor descs and check for errors 
    checkCUDAERRORS(cudnnCreateTensorDescriptor(&input_desc));
    checkCUDAERRORS(cudnnSetTensor4dDescriptor(input_desc,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/1,
        /*channels=*/3,
        /*image_height=*/image.rows,
        /*image_width=*/image.cols));

    checkCUDAERRORS(cudnnCreateFilterDescriptor(&filter_desc));
    checkCUDAERRORS(cudnnSetFilter4dDescriptor(filter_desc,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/3,
        /*in_channels=*/3,
        /*kernel_height=*/3,
        /*kernel_width=*/3));

    checkCUDAERRORS(cudnnCreateConvolutionDescriptor(&convolution_desc));
    checkCUDAERRORS(cudnnSetConvolution2dDescriptor(convolution_desc,
        /*pad_height=*/1,
        /*pad_width=*/1,
        /*vertical_stride=*/1,
        /*horizontal_stride=*/1,
        /*dilation_height=*/1,
        /*dilation_width=*/1,
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));

    int batch_size{ 0 }, channels{ 0 }, height{ 0 }, width{ 0 };
    checkCUDAERRORS(cudnnGetConvolution2dForwardOutputDim(convolution_desc,
        input_desc,
        filter_desc,
        &batch_size,
        &channels,
        &height,
        &width));

    std::cerr << "Image dimensions of Output Image: "
        << height << " x " << width << " x " << channels
        << std::endl;

    checkCUDAERRORS(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDAERRORS(cudnnSetTensor4dDescriptor(output_descriptor,
        /*order of params=*/CUDNN_TENSOR_NHWC,
        /*type of data=*/CUDNN_DATA_FLOAT,
        /*size of batch=*/1,
        /*num channels=*/3,
        /*img height=*/image.rows,
        /*img width=*/image.cols));

    int requested_algo_count = 0, returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t conv_fwd_results[100];

    cudnnConvolutionFwdAlgo_t algorithm_conv;
    checkCUDAERRORS(
        cudnnGetConvolutionForwardAlgorithm_v7(CUDNN_CONTEXT,
            input_desc,
            filter_desc,
            convolution_desc,
            output_descriptor,
            requested_algo_count, // (cudnnConvolutionFwdPreference_t)forward_algo,
            &returned_algo_count, // workspace_size_specify,
            conv_fwd_results));

    size_t workspace_bytes{ 0 };
    checkCUDAERRORS(cudnnGetConvolutionForwardWorkspaceSize(CUDNN_CONTEXT,
        input_desc,
        filter_desc,
        convolution_desc,
        output_descriptor,
        algorithm_conv,
        &workspace_bytes));

    assert(workspace_bytes > 0);

    void* d_workspace{ nullptr };
    cudaMalloc(&d_workspace, workspace_bytes);

    int image_bytes = batch_size * channels * height * width * sizeof(float);

    float* d_input{ nullptr };
    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

    float* d_output{ nullptr };
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    const float kernel_template[3][3] = {
      {1, 1, 1},
      {1, -8, 1},
      {1, 1, 1}
    };

    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }

    float* d_kernel{ nullptr };
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1.0f, beta = 0.0f;

    // do a forward propagation 
    checkCUDAERRORS(cudnnConvolutionForward(CUDNN_CONTEXT,
        &alpha,
        input_desc,
        d_input,
        filter_desc,
        d_kernel,
        convolution_desc,
        algorithm_conv,
        d_workspace,
        workspace_bytes,
        &beta,
        output_descriptor,
        d_output));

    // apply activation function if sigmoid_usage = true 
    if (sigmoid_usage) {
        cudnnActivationDescriptor_t activation_desc;
        checkCUDAERRORS(cudnnCreateActivationDescriptor(&activation_desc));
        checkCUDAERRORS(cudnnSetActivationDescriptor(activation_desc,
            CUDNN_ACTIVATION_SIGMOID,
            CUDNN_PROPAGATE_NAN,
            /*relu_coef=*/0));
        checkCUDAERRORS(cudnnActivationForward(CUDNN_CONTEXT,
            activation_desc,
            &alpha,
            output_descriptor,
            d_output,
            &beta,
            output_descriptor,
            d_output));
        cudnnDestroyActivationDescriptor(activation_desc);
    }

    float* h_output = new float[image_bytes];
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);

    SAVE_IMAGE("crack_out.png", h_output, height, width);

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(convolution_desc);

    cudnnDestroy(CUDNN_CONTEXT);

    std::cout << "Done!\n"; 
}


/*
Sources
https://github.com/BVLC/caffe/blob/master/src/caffe/layers/cudnn_conv_layer.cpp
http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/
*/

{% endhighlight %}

If you've run this correctly your input image, `crack.png`, should look like 
this:
![](/imgs/crack.jpg "Input")
<!-- <img src="../imgs/crack.jpg" alt="MarineGEO circle logo" style="height: 100px; width:100px;"/> -->


And your edge detected crack, `crack_out.png`, should look like this:
![Output](/imgs/crack_out.png)
<!-- <img src="../imgs/crack_out.png" alt="MarineGEO circle logo"/> -->