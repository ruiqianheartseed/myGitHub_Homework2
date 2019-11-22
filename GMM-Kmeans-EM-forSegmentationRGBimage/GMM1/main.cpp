//只可以分割8/24位的bmp图
//不可以输入其他位深度的图！！！
#include<stdio.h>
#include"GMM.h"
#include "KMeans.h"
#include"IEM2.h"
#include<iostream>
#include<windows.h>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<iomanip>
#include <algorithm>
#include<ctime>
#define BMPPIC "city4.bmp"
#define OUTPIC "result4-8k.bmp"
#define NKIND 8
using namespace std;
//读取、存储BMP位图文件
unsigned char *pBmpBuf;//读入图像数据的指针
int bmpWidth;//图像的宽
int bmpHeight;//图像的高
int lineByte;//每行字节数
RGBQUAD *pColorTable;//颜色表指针
int biBitCount;//图像类型，每像素位数

bool readBmp(char *bmpName)
{
	FILE *fp = fopen(bmpName, "rb");//二进制读方式打开指定的图像文件
	if (fp == 0)
		return 0;
	//跳过位图文件头结构BITMAPFILEHEADER
	fseek(fp, sizeof(BITMAPFILEHEADER), 0);
	//定义位图信息头结构变量，读取位图信息头进内存，存放在变量head中
	BITMAPINFOHEADER head;
	fread(&head, sizeof(BITMAPINFOHEADER), 1, fp); //获取图像宽、高、每像素所占位数等信息
	bmpWidth = head.biWidth;
	bmpHeight = head.biHeight;
	biBitCount = head.biBitCount;//定义变量，计算图像每行像素所占的字节数（必须是4的倍数）
	lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;//灰度图像有颜色表，且颜色表表项为256
	if (biBitCount == 8)
	{
		//申请颜色表所需要的空间，读颜色表进内存
		pColorTable = new RGBQUAD[256];
		fread(pColorTable, sizeof(RGBQUAD), 256, fp);
		
	}
	//申请位图数据所需要的空间，读位图数据进内存
		pBmpBuf = new unsigned char[lineByte * bmpHeight];
		fread(pBmpBuf, 1, lineByte * bmpHeight, fp);
		fclose(fp);//关闭文件
		return 1;//读取文件成功
}

bool saveBmp(char *bmpName, unsigned char *imgBuf, int width, int height, int biBitCount, RGBQUAD *pColorTable)
{
	//如果位图数据指针为0，则没有数据传入，函数返回
	if (!imgBuf)
		return 0;
	//颜色表大小，以字节为单位，灰度图像颜色表为1024字节，彩色图像颜色表大小为0
	int colorTablesize = 0;
	if (biBitCount == 8)
		colorTablesize = 1024;
	//待存储图像数据每行字节数为4的倍数
	int lineByte = (width * biBitCount / 8 + 3) / 4 * 4;
	//以二进制写的方式打开文件
	FILE *fp = fopen(bmpName, "wb");
	if (fp == 0)
		return 0;
	//申请位图文件头结构变量，填写文件头信息
	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42;//bmp类型
	//bfSize是图像文件4个组成部分之和
	fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	//bfOffBits是图像文件前3个部分所需空间之和
	fileHead.bfOffBits = 54 + colorTablesize;
	//写文件头进文件
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
	//申请位图信息头结构变量，填写信息头信息
	BITMAPINFOHEADER head;
	head.biBitCount = biBitCount;
	head.biClrImportant = 0;
	head.biClrUsed = 0;
	head.biCompression = 0;
	head.biHeight = height;
	head.biPlanes = 1;
	head.biSize = 40;
	head.biSizeImage = lineByte * height;
	head.biWidth = width;
	head.biXPelsPerMeter = 0;
	head.biYPelsPerMeter = 0;
	//写位图信息头进内存
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	//如果灰度图像，有颜色表，写入文件 
	if (biBitCount == 8)
		fwrite(pColorTable, sizeof(RGBQUAD), 256, fp);
	//写位图数据进文件
	fwrite(imgBuf, height*lineByte, 1, fp);
	//关闭文件
	fclose(fp);
	return 1;//存储文件成功
}

int playGMM()
{
	char readPath[] = BMPPIC;
	char writePath[] = OUTPIC;
	readBmp(readPath);
	int nVec = bmpWidth * bmpHeight;   //数据点个数
	int nDim = 3;//数据的维数
    int nPat = NKIND; //聚类个数
	if (biBitCount == 8) {
		int nDim = 1; 
	}
	    
	int col = 128 / nPat;
	GMM gmm(nVec, nDim, nPat);
	double **a = new double*[nVec];
	for (int i = 0; i < nVec; i++)
	{
		a[i] = new double[nDim];
	}
	for (int i = 0; i < bmpWidth; i++) {
		for (int j = 0; j < bmpHeight; j++) {
			for (int k = 0; k < nDim; k++) {
				a[bmpWidth*j + i][k] = pBmpBuf[lineByte*j + nDim*i + k];
			}
		}
	}
	gmm.Cluster(a);
	for (int i = 0; i < bmpWidth; i++) {
		for (int j = 0; j < bmpHeight; j++) {
			if (nDim == 1) 
			{
				double *temp = new double[nPat];
				for (int k = 0; k < nPat; k++) temp[k] = gmm.labelPro[bmpWidth*j + i][k];
				sort(temp, temp + nPat);
				for (int k = 0; k < nPat; k++)
				{
					if (gmm.labelPro[bmpWidth*j + i][k] == temp[0]) pBmpBuf[lineByte*j +  i] = 255/nPat*k;
				}
			}
			else 
			{
				double *temp = new double[nPat];
				for (int k = 0; k < nPat; k++) temp[k] = gmm.labelPro[bmpWidth*j + i][k];
				sort(temp, temp + nPat);
				for (int k = 0; k < nPat; k++) 
				{
					if (gmm.labelPro[bmpWidth*j + i][k] == temp[nPat-1]) 
					{
						pBmpBuf[lineByte*j + 3 * i + 2] =((k%3)*(255-2*k*col))%255;//R//128+k*127 ;//
						pBmpBuf[lineByte*j + 3 * i + 1] =(((k+1)%3)*(255-k*col))%255;//G// 0 + k*255;// 
						pBmpBuf[lineByte*j + 3 * i + 0] =(((k+2)%3)*(128+k*col))%255;//B //255-k*255 ;// 
					}
				
				}
			}
		}
	}
	saveBmp(writePath, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	delete[]pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
	return 0;
}

int playKMeans()
{
	char readPath[] = BMPPIC;
	char writePath[] = OUTPIC;
	readBmp(readPath);
	int nVec = bmpWidth * bmpHeight;   //数据点个数
	int nDim = 3;//数据的维数
	int nPat = NKIND; //聚类个数
	if (biBitCount == 8) {
		int nDim = 1;
	}
	
	KMeans* kmeans = new KMeans(nDim, nPat);
	int* labels = new int[nVec];
	kmeans->SetInitMode(KMeans::InitUniform);
	double *data;
	data = new double [nVec*nDim];

		for (int i = 0; i < bmpWidth; i++) {
			for (int j = 0; j < bmpHeight; j++) {
				for (int k = 0; k < nDim; k++) {
					data[(bmpWidth*j+ i)*nDim + k]= pBmpBuf[lineByte*j + nDim * i + k];
				}
			}
		}
	
	kmeans->Cluster(data, nVec, labels);

	//输出pBmpBuf染色
	int col = 128 / nPat;
	for (int i = 0; i < bmpWidth; i++)
	{
		for (int j = 0; j < bmpHeight; j++)
		{
			int place = j * bmpWidth + i;
			int k = labels[place];

			if (nDim == 1) pBmpBuf[lineByte*j + i] = 255 / nPat * k;
			
			else
			{
				pBmpBuf[lineByte*j + 3 * i + 2] = ((k % 3)*(255 - 2 * k*col)) % 255;//R//128+k*127 ;//
				pBmpBuf[lineByte*j + 3 * i + 1] = (((k + 1) % 3)*(255 - k * col)) % 255;//G// 0 + k*255;// 
				pBmpBuf[lineByte*j + 3 * i + 0] = (((k + 2) % 3)*(128 + k * col)) % 255;//B //255-k*255 ;// 
			}
		}
	}
	saveBmp(writePath, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	delete[]data;
	delete[]pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
	return 0;
}

int playBIEM()
{
	char readPath[] = BMPPIC;
	char writePath[] = OUTPIC;
	readBmp(readPath);
	int nVec = bmpWidth * bmpHeight;   //数据点个数
	int nDim = 3;//数据的维数
	int nPat = NKIND; //聚类个数
	if (biBitCount == 8) {
		int nDim = 1;
	}

	int col = 128 / nPat;
	double **a = new double*[nVec];
	for (int i = 0; i < nVec; i++)
	{
		a[i] = new double[nDim];
	}
	for (int i = 0; i < bmpWidth; i++) {
		for (int j = 0; j < bmpHeight; j++) {
			for (int k = 0; k < nDim; k++) {
				a[bmpWidth*j + i][k] = pBmpBuf[lineByte*j + nDim * i + k];
			}
		}
	}

	IEM2 gmm(nVec, nDim, nPat,a);
	gmm.Cluster(a);
	for (int i = 0; i < bmpWidth; i++) {
		for (int j = 0; j < bmpHeight; j++) {
			if (nDim == 1)
			{
				double *temp = new double[nPat];
				for (int k = 0; k < nPat; k++) temp[k] = gmm.labelPro[bmpWidth*j + i][k];
				sort(temp, temp + nPat);
				for (int k = 0; k < nPat; k++)
				{
					if (gmm.labelPro[bmpWidth*j + i][k] == temp[0]) pBmpBuf[lineByte*j + i] = 255 / nPat * k;
				}
			}
			else
			{
				double *temp = new double[nPat];
				for (int k = 0; k < nPat; k++) temp[k] = gmm.labelPro[bmpWidth*j + i][k];
				sort(temp, temp + nPat);
				for (int k = 0; k < nPat; k++)
				{
					if (gmm.labelPro[bmpWidth*j + i][k] == temp[nPat - 1])
					{
						pBmpBuf[lineByte*j + 3 * i + 2] = ((k % 3)*(255 - 2 * k*col)) % 255;//R//128+k*127 ;//
						pBmpBuf[lineByte*j + 3 * i + 1] = (((k + 1) % 3)*(255 - k * col)) % 255;//G// 0 + k*255;// 
						pBmpBuf[lineByte*j + 3 * i + 0] = (((k + 2) % 3)*(128 + k * col)) % 255;//B //255-k*255 ;// 
					}

				}
			}
		}
	}
	saveBmp(writePath, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	delete[]pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
	return 0;
}

int playIEM()
{
	char readPath[] = BMPPIC;
	char writePath[] = OUTPIC;
	readBmp(readPath);
	int nVec = bmpWidth * bmpHeight;   //数据点个数
	int nDim = 3;//数据的维数
	int nPat = NKIND; //聚类个数
	if (biBitCount == 8) {
		int nDim = 1;
	}

	int col = 128 / nPat;
	double **a = new double*[nVec];
	for (int i = 0; i < nVec; i++)
	{
		a[i] = new double[nDim];
	}
	for (int i = 0; i < bmpWidth; i++) {
		for (int j = 0; j < bmpHeight; j++) {
			for (int k = 0; k < nDim; k++) {
				a[bmpWidth*j + i][k] = pBmpBuf[lineByte*j + nDim * i + k];
			}
		}
	}

	IEM2 gmm(nVec, nDim, nPat, a);
	gmm.Cluster2(a);
	for (int i = 0; i < bmpWidth; i++) {
		for (int j = 0; j < bmpHeight; j++) {
			if (nDim == 1)
			{
				double *temp = new double[nPat];
				for (int k = 0; k < nPat; k++) temp[k] = gmm.labelPro[bmpWidth*j + i][k];
				sort(temp, temp + nPat);
				for (int k = 0; k < nPat; k++)
				{
					if (gmm.labelPro[bmpWidth*j + i][k] == temp[0]) pBmpBuf[lineByte*j + i] = 255 / nPat * k;
				}
			}
			else
			{
				double *temp = new double[nPat];
				for (int k = 0; k < nPat; k++) temp[k] = gmm.labelPro[bmpWidth*j + i][k];
				sort(temp, temp + nPat);
				for (int k = 0; k < nPat; k++)
				{
					if (gmm.labelPro[bmpWidth*j + i][k] == temp[nPat - 1])
					{
						pBmpBuf[lineByte*j + 3 * i + 2] = ((k % 3)*(255 - 2 * k*col)) % 255;//R//128+k*127 ;//
						pBmpBuf[lineByte*j + 3 * i + 1] = (((k + 1) % 3)*(255 - k * col)) % 255;//G// 0 + k*255;// 
						pBmpBuf[lineByte*j + 3 * i + 0] = (((k + 2) % 3)*(128 + k * col)) % 255;//B //255-k*255 ;// 
					}

				}
			}
		}
	}
	saveBmp(writePath, pBmpBuf, bmpWidth, bmpHeight, biBitCount, pColorTable);
	delete[]pBmpBuf;
	if (biBitCount == 8)
		delete[]pColorTable;
	return 0;
}


int main() {
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	//请手动切换模式
	//playGMM();
	playKMeans();
	//playIEM();
	//playBIEM();//Batch-IEM
	endTime = clock();//计时结束
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;

}

