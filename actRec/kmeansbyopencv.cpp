#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <fstream>


using namespace cv;
using namespace std;

/*
	kmeans聚类
	mykmeans input.txt output.txt rows cols nClusters eps

*/


int main( int argc, char** argv )
{
	//1、加载数据
	//const int ROWS = 120000;
	//const int COLS = 225;
	int rows = atoi(argv[3]);
	int cols = atoi(argv[4]);
	int nClusters = atoi(argv[5]);
	double eps = atof(argv[6]);

	Mat p = Mat::zeros(rows,cols,CV_32F);
	ifstream fin;	//文件相关变量
	float a;		//存放临时变量

	
	cout<<"载入数据中。。。"<<endl;
	fin.open(argv[1]);
	//fin.open("data\\poolmax_nF225_bN5000_iter100.txt");
	if(!fin)
	{
		cout<<"无法打开文件"<<endl;
		return -1;
	}

	for(int row=0;row<rows;row++)
	{
		float* pdata = p.ptr<float>(row);
		for(int col=0;col<cols;col++)
		{
			fin>>a;
			pdata[col] = a;
		}
	}
	fin.close();
	cout<<"载入数据完毕！"<<endl;
	cout<<"显示第一行数据："<<endl;
	for(int i=0;i<1;i++)
	{
		for(int j=0;j<cols;j++)
		{
			cout<<p.at<float>(i,j)<<" ";
		}
		cout<<endl;
	}
	cout<<"*****************************************************"<<endl;

	cout<<"kmeans聚类中。。。"<<endl;
	//const int nClusters = 4000;
	Mat bestLabels, centers;
	//eps:0.5
	kmeans(p, nClusters, bestLabels,
        TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, eps),
        3, KMEANS_PP_CENTERS, centers);

	cout<<"kmeans聚类结束，保存bestLabels数组。。。"<<endl;
	cout<<"eps:"<<eps<<endl;

	ofstream fout;
	//fout.open("data\\poolmax_nF225_bN5000_iter100_bestLabels_c4000_EPS0_5.txt");
	fout.open(argv[2]);

	//fout<<bestLabels;
	//bestLabels是60000行一列的int矩阵
	for(int row=0;row<rows;row++)
	{
		fout<<bestLabels.at<int>(row,0)<<endl;
	}
	fout << flush; 
	fout.close();
	cout<<"保存bestLabels完毕！"<<endl;

	cout<<"前60个bestLabels："<<endl;
	for(int row=0;row<60;row++)
	{
		cout<<bestLabels.at<int>(row,0)<<" ";
	}
	cout<<endl;
	cout<<"保存bestLabels数组完毕！"<<endl;

	
	return 0;
}