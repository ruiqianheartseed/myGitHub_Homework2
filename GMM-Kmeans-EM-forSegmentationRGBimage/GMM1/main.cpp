//ֻ���Էָ�8/24λ��bmpͼ
//��������������λ��ȵ�ͼ������
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
//��ȡ���洢BMPλͼ�ļ�
unsigned char *pBmpBuf;//����ͼ�����ݵ�ָ��
int bmpWidth;//ͼ��Ŀ�
int bmpHeight;//ͼ��ĸ�
int lineByte;//ÿ���ֽ���
RGBQUAD *pColorTable;//��ɫ��ָ��
int biBitCount;//ͼ�����ͣ�ÿ����λ��

bool readBmp(char *bmpName)
{
	FILE *fp = fopen(bmpName, "rb");//�����ƶ���ʽ��ָ����ͼ���ļ�
	if (fp == 0)
		return 0;
	//����λͼ�ļ�ͷ�ṹBITMAPFILEHEADER
	fseek(fp, sizeof(BITMAPFILEHEADER), 0);
	//����λͼ��Ϣͷ�ṹ��������ȡλͼ��Ϣͷ���ڴ棬����ڱ���head��
	BITMAPINFOHEADER head;
	fread(&head, sizeof(BITMAPINFOHEADER), 1, fp); //��ȡͼ����ߡ�ÿ������ռλ������Ϣ
	bmpWidth = head.biWidth;
	bmpHeight = head.biHeight;
	biBitCount = head.biBitCount;//�������������ͼ��ÿ��������ռ���ֽ�����������4�ı�����
	lineByte = (bmpWidth * biBitCount / 8 + 3) / 4 * 4;//�Ҷ�ͼ������ɫ������ɫ�����Ϊ256
	if (biBitCount == 8)
	{
		//������ɫ������Ҫ�Ŀռ䣬����ɫ����ڴ�
		pColorTable = new RGBQUAD[256];
		fread(pColorTable, sizeof(RGBQUAD), 256, fp);
		
	}
	//����λͼ��������Ҫ�Ŀռ䣬��λͼ���ݽ��ڴ�
		pBmpBuf = new unsigned char[lineByte * bmpHeight];
		fread(pBmpBuf, 1, lineByte * bmpHeight, fp);
		fclose(fp);//�ر��ļ�
		return 1;//��ȡ�ļ��ɹ�
}

bool saveBmp(char *bmpName, unsigned char *imgBuf, int width, int height, int biBitCount, RGBQUAD *pColorTable)
{
	//���λͼ����ָ��Ϊ0����û�����ݴ��룬��������
	if (!imgBuf)
		return 0;
	//��ɫ���С�����ֽ�Ϊ��λ���Ҷ�ͼ����ɫ��Ϊ1024�ֽڣ���ɫͼ����ɫ���СΪ0
	int colorTablesize = 0;
	if (biBitCount == 8)
		colorTablesize = 1024;
	//���洢ͼ������ÿ���ֽ���Ϊ4�ı���
	int lineByte = (width * biBitCount / 8 + 3) / 4 * 4;
	//�Զ�����д�ķ�ʽ���ļ�
	FILE *fp = fopen(bmpName, "wb");
	if (fp == 0)
		return 0;
	//����λͼ�ļ�ͷ�ṹ��������д�ļ�ͷ��Ϣ
	BITMAPFILEHEADER fileHead;
	fileHead.bfType = 0x4D42;//bmp����
	//bfSize��ͼ���ļ�4����ɲ���֮��
	fileHead.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + colorTablesize + lineByte * height;
	fileHead.bfReserved1 = 0;
	fileHead.bfReserved2 = 0;
	//bfOffBits��ͼ���ļ�ǰ3����������ռ�֮��
	fileHead.bfOffBits = 54 + colorTablesize;
	//д�ļ�ͷ���ļ�
	fwrite(&fileHead, sizeof(BITMAPFILEHEADER), 1, fp);
	//����λͼ��Ϣͷ�ṹ��������д��Ϣͷ��Ϣ
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
	//дλͼ��Ϣͷ���ڴ�
	fwrite(&head, sizeof(BITMAPINFOHEADER), 1, fp);
	//����Ҷ�ͼ������ɫ��д���ļ� 
	if (biBitCount == 8)
		fwrite(pColorTable, sizeof(RGBQUAD), 256, fp);
	//дλͼ���ݽ��ļ�
	fwrite(imgBuf, height*lineByte, 1, fp);
	//�ر��ļ�
	fclose(fp);
	return 1;//�洢�ļ��ɹ�
}

int playGMM()
{
	char readPath[] = BMPPIC;
	char writePath[] = OUTPIC;
	readBmp(readPath);
	int nVec = bmpWidth * bmpHeight;   //���ݵ����
	int nDim = 3;//���ݵ�ά��
    int nPat = NKIND; //�������
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
	int nVec = bmpWidth * bmpHeight;   //���ݵ����
	int nDim = 3;//���ݵ�ά��
	int nPat = NKIND; //�������
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

	//���pBmpBufȾɫ
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
	int nVec = bmpWidth * bmpHeight;   //���ݵ����
	int nDim = 3;//���ݵ�ά��
	int nPat = NKIND; //�������
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
	int nVec = bmpWidth * bmpHeight;   //���ݵ����
	int nDim = 3;//���ݵ�ά��
	int nPat = NKIND; //�������
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
	startTime = clock();//��ʱ��ʼ
	//���ֶ��л�ģʽ
	//playGMM();
	playKMeans();
	//playIEM();
	//playBIEM();//Batch-IEM
	endTime = clock();//��ʱ����
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	system("pause");
	return 0;

}

