#pragma once
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <ctype.h>


#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <string>

IplImage* image = 0; 
IplImage* prev_image = 0;
CvCapture* capture = 0;

int show = 0; 

int videotest( char* video )
{
	int frameNum = 0;

	//char* video = argv[1];
	capture = cvCreateFileCapture(video);

	if( !capture ) { 
		printf( "Could not initialize capturing..\n" );
		return -1;
	}
	
	if( show == 1 )
		cvNamedWindow( "Video", 0 );

	while( true ) {
		IplImage* frame = 0;
		int i, j, c;

		// get a new frame
		frame = cvQueryFrame( capture );
		if( !frame )
			break;

		if( !image ) {
			image =  cvCreateImage( cvSize(frame->width,frame->height), 8, 3 );
			image->origin = frame->origin;
		}

		cvCopy( frame, image, 0 );

		if( show == 1 ) {
			cvShowImage( "Video", image);
			c = cvWaitKey(33);
			if((char)c == 27) break;
		}
		
		//std::cerr << "The " << frameNum << "-th frame" << std::endl;
		frameNum++;
	}

	if( show == 1 )
		cvDestroyWindow("Video");

	return 0;
}

int videotest2(char* video)
{
	VideoCapture capture;

	//char* video ="D:\\KTH\\demo\\person01_boxing_d1_uncomp.avi";
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int fcnt=0;
	while(true) 
	{
		Mat frame;
		int i; 
		char c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		fcnt++;
	}
	return fcnt;

}

//批量测试youtube数据是否OK
int test_youtube()
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
	fout.open("H:\\action_recognition\\code\\data\\densefeature_1\\youtube\\youtube_videotest.txt");
	char* youtube_dir = "H:\\action_recognition\\dataset\\youtube\\";
	char* parent_dir="H:\\action_recognition\\code\\data\\densefeature_1\\youtube\\";
	for(int cc=0;cc<classnum;cc++)
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
				
				//int res = feature_extract(path,trapospath,trajectorypath,hogpath,hofpath,mbhxpath,mbhypath);
				//cout<<"traceNum:"<<res<<endl;
				//fout<<res<<endl;
				int res = videotest2(path);
				cout<<res<<endl;
				fout<<res<<endl;
			}
		}
		
	}

	fout.close();
	return 0;
}