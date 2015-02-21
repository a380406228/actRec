#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include "kth_extract.h"

using namespace cv;
using namespace std;

/*
	根据data数据和KTH原始视频提取patches原始信息

*/


/*
	批量
*/


int main( int argc, char** argv )
{
	//extract("D:\\KTH\\boxing\\person01_boxing_d1_uncomp.avi","D:\\myproject\\data\\datarand200\\person01_boxing_d1_pos.txt","data\\person01_boxing_d1_origin.txt");
    //return 0;
	
	int dnum=4;
	int actnum=6;
	int personnum=25;
	char* actname[6]={"boxing","handclapping","handwaving","jogging","running","walking"};

	/*
	for di=1:dnum
    for p=1:personnum
        shi = floor(p/10);
        ge = mod(p,10);
        for i=1:actnum
            cnt=cnt+1;
            path = sprintf('person%d%d_%s_d%d_pos.txt',shi,ge,actname{i},di);
            disp(path);
	*/

	//for test
	//dnum=1;
	//actnum=1;
	//personnum=1;
	
	int proscnt=0;
	for(int di=1;di<=dnum;di++)
	{
		for(int p=1; p<=personnum; p++)
		{
			int shi = p/10;
			int ge = p%10;
			for(int i=0;i<actnum;i++)
			{
				char videopath[100];
				char datapath[100];
				char outpath[100];
				sprintf(videopath,"D:\\KTH\\%s\\person%d%d_%s_d%d_uncomp.avi",actname[i],shi,ge,actname[i],di);
				sprintf(datapath,"D:\\myproject\\data\\datarand200\\person%d%d_%s_d%d_pos.txt",shi,ge,actname[i],di);
				sprintf(outpath,"D:\\myproject\\data\\dataoriginrand200\\person%d%d_%s_d%d_origin.txt",shi,ge,actname[i],di);
				cout<<"videopath:"<<videopath<<endl;
				cout<<"datapath:"<<datapath<<endl;
				cout<<"outpath:"<<outpath<<endl;

				extract(videopath,datapath,outpath);
				proscnt++;
				cout<<"***pros:"<<proscnt<<endl;
			}
		}
	}

	return 0;
}