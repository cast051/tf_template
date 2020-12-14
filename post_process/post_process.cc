#include <stdio.h>
#include <cfloat>
#include <list>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/work_sharder.h"

using namespace tensorflow;
using namespace cv;
using namespace std;
typedef Eigen::ThreadPoolDevice CPUDevice;
/*
 * mask:[H,W] (int8/uint8)
 * output:[X,3] {(x,y,type),....}
 */
REGISTER_OP("Seg2PointNum")
    // .Attr("T: uint8")
    .Input("mask: uint8")
	.Output("output:int32")
    .SetShapeFn([](shape_inference::InferenceContext* c){
        c->set_output(0,c->MakeShape({-1,-1,3 }));
//        c->set_output(0,c->input(0));
		return Status::OK();
    });

// template <typename Device, typename T>
class Seg2PointNumOp: public OpKernel {
	public:
		explicit Seg2PointNumOp(OpKernelConstruction* context) : OpKernel(context) {
		}
		void Compute(OpKernelContext* context) override
        {
            const Tensor& mask_tensor = context->input(0);
            auto mask = mask_tensor.flat<uint8>(); 
            const auto    batch_size = mask_tensor.dim_size(0);
            const auto    height     = mask_tensor.dim_size(1);
            const auto    width      = mask_tensor.dim_size(2);
            vector<vector<Point2d>> cell_point_b;
            int size_=0;
            for(int i=0;i<batch_size;++i)
            {
                Mat mask_Mat=Mat(height,width,CV_8UC1,(void*)(mask.data()+width*height*i));
                vector<Point2d> cell_point=Mask2PointNum(mask_Mat);
                if(size_<cell_point.size())
                {
                    size_=cell_point.size();
                }
                cell_point_b.push_back(cell_point);
            }

            Tensor* output_tensor = NULL;
            TensorShape output_shape={batch_size,size_,3};
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,&output_tensor));
            auto tensor = output_tensor->template tensor<int,3>();
            tensor.setZero();
            for(int i=0;i<batch_size;++i)
            {
                for(int j=0; j<cell_point_b[i].size(); ++j) {
                            auto& p = cell_point_b[i][j];
                        tensor(i,j,0) = p.x;
                        tensor(i,j,1) = p.y;
                        tensor(i,j,2) = 1;
                }
            }
        }


        static vector<Point2d> Mask2PointNum(Mat mask)
        {
            float  point_area=78.5;
            float area_para=0.5;
            Mat imageShold;
            threshold(mask, imageShold, 50, 255, THRESH_BINARY);//二值化
            Mat element = getStructuringElement(MORPH_RECT,Size(5, 5));
            Mat morphology;
            morphologyEx(imageShold, morphology, MORPH_CLOSE, element);

            Mat labels = Mat::zeros(mask.size(), CV_16U);
            Mat stats, centroids;
            int n_comps = connectedComponentsWithStats(morphology, labels, stats, centroids, 4);//nccomps是轮廓个数
            //int n_comps = connectedComponents(imageShold, labels, 4, CV_16U);
            n_comps -= 1;
            int blend_count = 0;
            int discard_count = 0;
            vector<Point2d> cell_point;
            for (int i = 1; i < stats.rows; i++)
            {
                int area = stats.at<int>(i, 4);
                if (area > area_para*point_area)
                {
                    Point2d point_;
                    point_.x = stats.at<int>(i, 0)+ stats.at<int>(i, 2) / 2;
                    point_.y = stats.at<int>(i, 1) + stats.at<int>(i, 3) / 2;
                    cell_point.push_back(point_);
                }
                else 
                {
                    discard_count += 1;
                }
            }
            // cout << "连通域个数" << n_comps << endl;
            // cout << "排除的个数" << n_comps- cell_point.size() << endl;
            // cout << "size :" << cell_point.size() << endl;
            return cell_point;
        }
};
REGISTER_KERNEL_BUILDER(Name("Seg2PointNum").Device(DEVICE_CPU), Seg2PointNumOp);
