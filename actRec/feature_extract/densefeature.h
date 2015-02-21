#pragma once

/*
	KTH视频库的HOG、HOF、MBH、trajectory和pos特征批量提取
	KTH视频库包含：6类动作，每类100个样本
	最大轨迹上限：5000 （有可能不到这么多）

*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "DenseTrack.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

using namespace cv;
using namespace std;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories
const int MAXTRACENUM = 5000;

//把HOG、HOF、MBH输出到文件
void PrintDescToFile(std::vector<float>& desc, DescInfo& descInfo, TrackInfo& trackInfo,const char* filename)
{
	ofstream fout;
	fout.open(filename,ios::app);
	int tStride = cvFloor(trackInfo.length/descInfo.ntCells);
	float norm = (float)(1./float(tStride));
	int dim = descInfo.dim;
	int pos = 0;
	for(int i = 0; i < descInfo.ntCells; i++) 
	{
		std::vector<float> vec(dim);
		for(int t = 0; t < tStride; t++)
			for(int j = 0; j < dim; j++)
				vec[j] += desc[pos++];
		for(int j = 0; j < dim; j++)
			//printf("%.7f\t", vec[j]*norm);
			fout<<vec[j]*norm<<" ";
	}
	fout<<endl;
	fout.close();
}


/*
	提取轨迹特征，并保存到txt文件中
	MAXTRACENUM： 设置轨迹条数的上限
	trapath： 原始的轨迹坐标，包括起点坐标，一共16*2=32 float
	trajectorypath：轨迹偏移量
*/
int feature_extract(char* video,char* trapospath,char* trajectorypath, char* hogpath,char* hofpath,char* mbhxpath,char* mbhypath)
{
	int res = 0;
	ofstream fout,fout1;
	//fout.open("save\\msg.txt");
	fout.open(trapospath);
	fout1.open(trajectorypath);

	VideoCapture capture;

	//char* video ="D:\\KTH\\demo\\person01_boxing_d1_uncomp.avi";
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int validtracecnt = 0;
	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);
	//输出视频帧总长度
	cout<<"TotalLen:"<<seqInfo.length<<endl;
	//fout_msg<<"TotalLen:"<<seqInfo.length<<endl;
	//if(flag)
		//seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrack", 0);

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0); // for optical flow

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	while(true) {
		Mat frame;
		int i; 
		char c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		//输出帧数,每隔10帧输出一次
		if(frame_num%10 == 0)
		{
			cout<<frame_num<<" ";
			//fout_msg<<frame_num<<" ";
		}

		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);

			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion 计算多项式展开
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

				// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) 
				{
					
					std::vector<Point2f> trajectory(trackInfo.length+1);
					std::vector<Point2f> trajectory2(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
					{
						trajectory[i] = iTrack->point[i]*fscales[iScale];
						trajectory2[i] = iTrack->point[i]*fscales[iScale];
					}
				
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					//float start_ptx,start_pty;
					
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length))
					{
						res = validtracecnt;
						validtracecnt++;
						if(validtracecnt > MAXTRACENUM)
						{					
							goto END;
						}
						
						/*
						printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

						// for spatio-temporal pyramid
						printf("%f\t", std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999));
						printf("%f\t", std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999));
					
						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
							printf("%f\t%f\t", trajectory[i].x,trajectory[i].y);
		
						PrintDesc(iTrack->hog, hogInfo, trackInfo);
						PrintDesc(iTrack->hof, hofInfo, trackInfo);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						printf("\n");
						*/
						/*
						//1、帧数(the last)
						fout<<frame_num<<" ";
						//2、输出比例系数
						fout<<fscales[iScale]<<" ";
						//3、轨迹起点x,y坐标和对应的rect.x,rect.y
						fout<<rect.x<<" "<<rect.y<<" "<<point.x<<" "<<point.y<<" ";
						//4、输出轨迹偏移量
						for (int i = 0; i < trackInfo.length; ++i)
							fout<<trajectory[i].x<<" "<<trajectory[i].y<<" ";
						fout<<mean_x<<" "<<mean_y<<" "<<var_x<<" "<<var_y<<" "<<length<<" ";
						fout<<std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999)<<" ";
						fout<<std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999)<<" ";
						fout<<std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999)<<" ";
						fout<<endl;
						//cout<<"seqInfo.length:"<<float(seqInfo.length)<<endl;

						PrintDescToFile(iTrack->hog, hogInfo, trackInfo, hogpath);
						PrintDescToFile(iTrack->hof, hofInfo, trackInfo, hofpath);
						PrintDescToFile(iTrack->mbhX, mbhInfo, trackInfo,mbhxpath);
						PrintDescToFile(iTrack->mbhY, mbhInfo, trackInfo,mbhypath);				
						*/

						//1、输出轨迹起始点、轨迹位置,未归一化并包含起始点，长度trackInfo.length（已经将scale计算在内）
						fout<<frame_num<<" "<<fscales[iScale]<<" ";
						for(int i = 0; i < trackInfo.length; ++i)
						{
							fout<<trajectory2[i].x<<" "<<trajectory2[i].y<<" ";
						}							
						fout<<endl;

						//fout1 trajectory msg
						for (int i = 0; i < trackInfo.length; ++i)
							fout1<<trajectory[i].x<<" "<<trajectory[i].y<<" ";
						fout1<<frame_num;
						fout1<<endl;

						//HOG HOF MBHX MBHY
						PrintDescToFile(iTrack->hog, hogInfo, trackInfo, hogpath);
						PrintDescToFile(iTrack->hof, hofInfo, trackInfo, hofpath);
						PrintDescToFile(iTrack->mbhX, mbhInfo, trackInfo,mbhxpath);
						PrintDescToFile(iTrack->mbhY, mbhInfo, trackInfo,mbhypath);		
					}
					

					iTrack = tracks.erase(iTrack);
					continue;
					

				}
				++iTrack;
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every initGap frames
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(unsigned int i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		frame_num++;

		if( show_track == 1 ) 
		{
			imshow( "DenseTrack", image);
			c = cvWaitKey(40);
			if((char)c == 27) break;
		}
	}

END:if( show_track == 1 )
		destroyWindow("DenseTrack");

	fout.close();
	fout1.close();
	return res;
}

//step1:批量提取KTH数据库的轨迹特征
int feature_extract_batch()
{
	char path[100];
	char *clsall[6] = {"boxing","handclapping","handwaving","jogging","running","walking"};
	char cls[20];
	//char* tracepath,char* hogpath,char* hofpath,char* mbhxpath,char* mbhypath
	char trapospath[100];
	char trajectorypath[100];
	char hogpath[100];
	char hofpath[100];
	char mbhxpath[100];
	char mbhypath[100];

	ofstream fout;
	fout.open("H:\\action_recognition\\code\\data\\densefeature_1\\kth\\feature_ext_msg.txt");
	char* kth_dir = "H:\\action_recognition\\dataset\\KTH\\";
	char* parent_dir="H:\\action_recognition\\code\\data\\densefeature_1\\kth\\";
	for(int cc=0;cc<6;cc++)
	{
		cout<<"\n******正在处理第"<<cc+1<<"类:"<<clsall[cc]<<"******"<<endl;
		fout<<"\n******正在处理第"<<cc+1<<"类:"<<clsall[cc]<<"******"<<endl;
		sprintf(cls,clsall[cc]);

		for(int i=1;i<=25;i++)
		{
			for(int j=1;j<=4;j++)
			{
				int shi = i/10;
				int ge = i%10;
				
				sprintf(path,"%s%s\\person%d%d_%s_d%d_uncomp.avi",kth_dir,cls,shi,ge,cls,j);
				cout<<"\n"<<path<<endl;
				fout<<"\n"<<path<<endl;

				sprintf(trapospath,		"%sperson%d%d_%s_d%d_pos.txt",		parent_dir,shi,ge,cls,j);
				sprintf(trajectorypath,	"%sperson%d%d_%s_d%d_trace.txt",	parent_dir,shi,ge,cls,j);
				sprintf(hogpath,			"%sperson%d%d_%s_d%d_hog.txt",		parent_dir,shi,ge,cls,j);
				sprintf(hofpath,			"%sperson%d%d_%s_d%d_hof.txt",		parent_dir,shi,ge,cls,j);
				sprintf(mbhxpath,		"%sperson%d%d_%s_d%d_mbhx.txt",	parent_dir,shi,ge,cls,j);
				sprintf(mbhypath,		"%sperson%d%d_%s_d%d_mbhy.txt",	parent_dir,shi,ge,cls,j);

				int res = feature_extract(path,trapospath,trajectorypath,hogpath,hofpath,mbhxpath,mbhypath);
				cout<<"traceNum:"<<res<<endl;
				fout<<res<<endl;
			}
		}
	}

	fout.close();
	return 0;
}

//step1_2:批量提取youtube数据库的轨迹特征
int feature_extract_batch_youtube()
{
	const int classnum = 11; //一共11类样本
	const int groupnum = 25;//每一类下有25个文件组
	const int spnum = 4;	//每个组下取4个样本
	//第一层目录名字
	//char *clsall_1[classnum] = {"basketball","biking","diving","golf_swing","horse_riding","soccer_juggling","swing","tennis_swing","trampoline_jumping","volleyball_spiking","walking" };
	//第二层目录名字
	char *clsall[classnum] = {"shooting","biking","diving","golf","riding","juggle","swing","tennis","jumping","spiking","walk_dog" };
	
	char path[200];
	char clsname[100];
	//char* tracepath,char* hogpath,char* hofpath,char* mbhxpath,char* mbhypath
	char trapospath[200];
	char trajectorypath[200];
	char hogpath[200];
	char hofpath[200];
	char mbhxpath[200];
	char mbhypath[200];

	ofstream fout;
	fout.open("H:\\action_recognition\\code\\data\\densefeature_1\\youtube\\youtube_feature_ext_msg.txt");
	char* youtube_dir = "H:\\action_recognition\\dataset\\youtube\\";
	char* parent_dir="H:\\action_recognition\\code\\data\\densefeature_1\\youtube\\";
	//cc=0,1,2, finished
	//cc=3,4,5,finished
	//cc = 6,7,8 finished
	//cc = 9,10
	for(int cc=9;cc<classnum;cc++)
	{
		cout<<"\n******正在处理第"<<cc+1<<"类:"<<clsall[cc]<<"******"<<endl;
		fout<<"\n******正在处理第"<<cc+1<<"类:"<<clsall[cc]<<"******"<<endl;
		//sprintf(clsname,clsall_1[cc]);
		for(int i=1;i<=groupnum;i++)
		{
			for(int j=1;j<=spnum;j++)
			{
				int shi = i/10;
				int ge = i%10;
				
				//读入视频文件
				sprintf(path,"%sv_%s_%d%d_0%d.avi",youtube_dir,clsall[cc],shi,ge,j);
				cout<<"\n"<<path<<endl;
				fout<<"\n"<<path<<endl;

				//指定输出路径 v_shooting_01_01.txt
				sprintf(trapospath,		"%sv_%s_%d%d_0%d_pos.txt",			parent_dir,clsall[cc],shi,ge,j);
				sprintf(trajectorypath,	"%sv_%s_%d%d_0%d_trace.txt",		parent_dir,clsall[cc],shi,ge,j);
				sprintf(hogpath,			"%sv_%s_%d%d_0%d_hog.txt",			parent_dir,clsall[cc],shi,ge,j);
				sprintf(hofpath,			"%sv_%s_%d%d_0%d_hof.txt",			parent_dir,clsall[cc],shi,ge,j);
				sprintf(mbhxpath,		"%sv_%s_%d%d_0%d_mbhx.txt",		parent_dir,clsall[cc],shi,ge,j);
				sprintf(mbhypath,		"%sv_%s_%d%d_0%d_mbhy.txt",		parent_dir,clsall[cc],shi,ge,j);
				
				int res = feature_extract(path,trapospath,trajectorypath,hogpath,hofpath,mbhxpath,mbhypath);
				cout<<"traceNum:"<<res<<endl;
				fout<<res<<endl;

			}
		}
		
	}

	fout.close();
	return 0;
}
