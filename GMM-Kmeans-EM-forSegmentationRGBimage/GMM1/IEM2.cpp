#include "IEM2.h"
#include "memory.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#define d 10 //增量步
#define nd 200 //预先训练
#define erro 0.1 //允许误差
#define randInitial 0
#define PI 3.1415926
#define CONST_E 0.000000000001//一个很小的数，用来解决协方差矩阵行列式为0以及高斯密度分布函数为0的问题
//IEM2的核心是：更新参数只使用每次一部分的像素，但是最后是要预测所有像素的标签的

//IEM2类输入输出结构参数以及初始化
IEM2::IEM2(int tnSample, int tnDimension, int tnLabel, double **iclsSample)
{
	allSample = iclsSample;//整体待分类的向量，最后只输出labelPro
	nAllSample = tnSample;//全部像素个数
	nDimension = tnDimension;
	nLabel = tnLabel;

	//labelPro[i][j]表示第i个向量属于第j类的概率,初始化为0，最后会输出
	labelPro = new double *[nAllSample];
	for (int i = 0; i < nAllSample; i++)//像素个数
	{
		labelPro[i] = new double[nLabel];
		memset(labelPro[i], 0, sizeof(double) * nLabel);
	}

	//labelU[i]表示第i类的均值向量，其初始化的值通过随机到k个中心means的方法得到，或者固定得到
	labelU = new double *[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelU[i] = new double[nDimension];
	}
	srand((unsigned)time(NULL));
	rand();

	//labelU[nLabel]的初始化
	bool *bSign = new bool[nAllSample];
	memset(bSign, false, sizeof(bool) * nAllSample);//每一份像素置false
	if (randInitial == 1)
	{
		for (int i = 0; i < nLabel; i++)//对于每一类，随机roll一个点
		{
			int nPt;
			do {
				nPt = (int)(((double)rand() / (double)RAND_MAX) * (nAllSample - 0.1));
			} while (bSign[nPt]);//如果roll的点重复了就再roll一次直到不重复
			bSign[nPt] = true;//roll过就标记已访问
			memcpy(labelU[i], allSample[nPt], sizeof(double) * nDimension);
		}
	}
	else
	{
		for (int i = 0; i < nLabel; i++)
		{
			int nPt;
			nPt = (int)(nAllSample / nLabel * i + nAllSample / (2 * nLabel));//固定初始化,均匀取点
			bSign[nPt] = true;
			memcpy(labelU[i], allSample[nPt], sizeof(double) * nDimension);
		}
	}
	delete[]bSign;

	printf("初始聚类中心\n");
	for (int i = 0; i < nLabel; i++)
	{
		printf("pattern %d mean", i);
		for (int j = 0; j < nDimension; j++) printf("%12.5lf", labelU[i][j]);
		printf("\n");
	}
	printf("\n");

	//labelCOV[i]表示第i类的协方差矩阵,并把协方差矩阵初始化成单位阵
	labelCOV = new double **[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelCOV[i] = new double *[nDimension];
		for (int j = 0; j < nDimension; j++)
		{
			labelCOV[i][j] = new double[nDimension];//i类，每一类都是3*3的矩阵，一行一行初始化
			memset(labelCOV[i][j], 0, sizeof(double) * nDimension);//先把对应行全部化为0
			labelCOV[i][j][j] = 1;//然后再把对角线置1
		}
	}

	//labelPripro[i]表示第i类的先验概率,初始化为1 / _nPat
	labelPripro = new double[nLabel];
	for (int i = 0; i < nLabel; i++) labelPripro[i] = 1.0 / nLabel;
}

IEM2::~IEM2()
{
	for (int i = 0; i < nSample; i++) delete[]labelPro[i];
	delete[]labelPro;

	for (int i = 0; i < nLabel; i++) delete[]labelU[i];
	delete[]labelU;

	for (int i = 0; i < nLabel; i++)
	{
		for (int j = 0; j < nDimension; j++) delete[]labelCOV[i][j];
		delete[]labelCOV[i];
	}
	delete[]labelCOV;
	delete[]labelPripro;
}

//3*3矩阵高斯消元法求逆
double **IEM2::Inverse(double **iMat)
{
	//复制原矩阵到tmpMat
	double **tmpMat = new double *[nDimension];
	for (int i = 0; i < nDimension; i++)
	{
		tmpMat[i] = new double[nDimension];
		memcpy(tmpMat[i], iMat[i], sizeof(double) * nDimension);
	}

	//创建一个单位阵
	double **tmpI = new double *[nDimension];
	for (int i = 0; i < nDimension; i++)
	{
		tmpI[i] = new double[nDimension];
		memset(tmpI[i], 0, sizeof(double) * nDimension);
		tmpI[i][i] = 1;
	}

	//Gaussian消元法求逆矩阵--正向消元
	for (int i = 0; i < nDimension; i++)
	{
		double lfTmp = tmpMat[i][i];
		for (int j = 0; j < nDimension; j++)
		{
			tmpMat[i][j] /= lfTmp;
			tmpI[i][j] /= lfTmp;
		}

		for (int j = i + 1; j < nDimension; j++)
		{
			double lfTmp = -tmpMat[j][i];
			for (int k = 0; k < nDimension; k++)
			{
				tmpMat[j][k] += lfTmp * tmpMat[i][k];
				tmpI[j][k] += lfTmp * tmpI[i][k];
			}
		}
	}

	//Gaussian消元法求逆矩阵--逆向消元
	for (int i = nDimension - 1; i >= 0; i--)
		for (int j = i - 1; j >= 0; j--)
		{
			double lfTmp = -tmpMat[j][i];
			for (int k = 0; k < nDimension; k++)
			{
				tmpMat[j][k] += tmpMat[i][k] * lfTmp;
				tmpI[j][k] += tmpI[i][k] * lfTmp;
			}
		}

	//销毁临时空间
	for (int i = 0; i < nDimension; i++) delete[]tmpMat[i];
	delete[]tmpMat;

	return tmpI;
}
//求3*3行列式的值
double IEM2::Determinant(double **iMat)
{
	//复制原矩阵
	double **tmpMat = new double *[nDimension];
	for (int i = 0; i < nDimension; i++)
	{
		tmpMat[i] = new double[nDimension];
		memcpy(tmpMat[i], iMat[i], sizeof(double) * nDimension);
	}

	//Gaussian消元法--得到对角形式
	for (int i = 0; i < nDimension; i++)
	{
		for (int j = i + 1; j < nDimension; j++)
		{
			double lfTmp = -tmpMat[j][i] / tmpMat[i][i];
			for (int k = 0; k < nDimension; k++) tmpMat[j][k] += lfTmp * tmpMat[i][k];
		}
	}

	//计算对角线乘积
	double det = 1;
	for (int i = 0; i < nDimension; i++) det *= tmpMat[i][i];

	//销毁临时空间
	for (int i = 0; i < nDimension; i++) delete[]tmpMat[i];
	delete[]tmpMat;

	return det;
}
//求多维高斯分布密度函数，当前计算的像素/类型/协方差矩阵
double IEM2::Gaussian(double *clsSample, double *labelU, double **labelCOV)
{
	double *Tmp1 = new double[nDimension];
	double *Tmp2 = new double[nDimension];

	for (int i = 0; i < nDimension; i++) Tmp1[i] = clsSample[i] - labelU[i];
	memset(Tmp2, 0, sizeof(double) * nDimension);

	double **InvlabelCOV = Inverse(labelCOV);

	for (int i = 0; i < nDimension; i++)
		for (int j = 0; j < nDimension; j++)
			Tmp2[i] += Tmp1[j] * InvlabelCOV[j][i];

	double lfTmp1 = 0;
	for (int i = 0; i < nDimension; i++)
		lfTmp1 += Tmp2[i] * Tmp1[i];
	lfTmp1 /= -2;
	lfTmp1 = exp(lfTmp1);

	double det = Determinant(labelCOV);
	double lfTmp2 = 1 / sqrt(pow(2 * PI, nDimension) * det) * lfTmp1;

	for (int i = 0; i < nDimension; i++) delete[]InvlabelCOV[i];
	delete[]InvlabelCOV;
	delete[]Tmp1;
	delete[]Tmp2;

	if (lfTmp2 < CONST_E) lfTmp2 = CONST_E;
	return lfTmp2;
}
//E
void IEM2::Expectation()
{
	//E-Step
	LLLH = 0;
	for (int i = 0; i < nSample; i++)
	{
		double Sum = 0;//用于计算likelihood
		for (int j = 0; j < nLabel; j++)
		{
			double Tmp = 0;
			for (int l = 0; l < nLabel; l++)
			{
				double G = Gaussian(clsSample[i], labelU[l], labelCOV[l]);//高斯分布密度
				Tmp += labelPripro[l] * G;
			}
			double G = Gaussian(clsSample[i], labelU[j], labelCOV[j]);//总体高斯分布密度
			labelPro0[i][j] = labelPripro[j] * G / Tmp;
			if (nSample == nAllSample) labelPro[i][j] = labelPro0[i][j];//这里的labelPro是会被用来输出的,被动更新
			Sum += labelPripro[j] * G;//但是影响判别的只有sum
		}
		LLLH += log(Sum);
	}
}
//M
void IEM2::Maximization()
{
	//M-Step
	for (int j = 0; j < nLabel; j++)
	{
		double Tmp1 = 0;
		for (int i = 0; i < nSample; i++) Tmp1 += labelPro0[i][j];

		//最大化步骤--求_pplfU[j]
		memset(labelU[j], 0, sizeof(double) * nDimension);//又把j类中心置0了？

		for (int i = 0; i < nSample; i++)
			for (int k = 0; k < nDimension; k++)
				labelU[j][k] += labelPro0[i][j] * clsSample[i][k];
		for (int k = 0; k < nDimension; k++) labelU[j][k] /= Tmp1;

		//最大化步骤--求_ppplfDelta[j]
		for (int i = 0; i < nDimension; i++)
			memset(labelCOV[j][i], 0, sizeof(double) * nDimension);
		for (int i = 0; i < nSample; i++)
		{
			double *Tmp2 = new double[nDimension];
			for (int a = 0; a < nDimension; a++) Tmp2[a] = clsSample[i][a] - labelU[j][a];
			for (int a = 0; a < nDimension; a++)
				labelCOV[j][a][a] += labelPro0[i][j] * Tmp2[a] * Tmp2[a];
			delete[]Tmp2;
		}
		for (int a = 0; a < nDimension; a++)
		{
			labelCOV[j][a][a] = labelCOV[j][a][a] / Tmp1;
			labelCOV[j][a][a] += CONST_E;
		}

		//最大化步骤--求_plfPi[j]
		labelPripro[j] = Tmp1 / nSample;
	}
}

void IEM2::Cluster(double **iclsSample)
{
	int Count = 0;
	int batch = nAllSample / d;
	nSample = batch;//当前在训练的像素个数
	//labelPro0[i][j]表示训练时i个向量属于第j类的概率
	labelPro0 = new double *[nSample];
	for (int j = 0; j < nSample; j++)//训练像素个数
	{
		labelPro0[j] = new double[nLabel];
		memset(labelPro0[j], 0, sizeof(double) * nLabel);
	}
	//要初始化clsSample
	clsSample = new double*[nSample];
	for (int ii = 0; ii < nSample; ii++)
	{
		clsSample[ii] = new double[nDimension];
	}
	srand((unsigned)time(NULL));
	rand();
	bool *bSign = new bool[nAllSample];

	//在前nd轮样本数都是不增加的
	for (int i = 0; i < nd; i++) {
		memset(bSign, false, sizeof(bool) * nAllSample);//每一份像素置false
		for (int k = 0; k < batch; k++) {
			int nPt;
			do {
				nPt = (int)(((double)rand() / (double)RAND_MAX) * (nAllSample - 0.1));
			} while (bSign[nPt]);//如果roll的点重复了就再roll一次直到不重复
			bSign[nPt] = true;//roll过就标记已访问
			for (int kk = 0; kk < nDimension; kk++)
			{
				clsSample[k][kk] = allSample[nPt][kk];
			}
		}		
		Expectation();
		Maximization();
		Count++;
		printf("样本数%d 第%d轮：%12.5lf\n", nSample, Count, LLLH *(double)(d));
	}

	delete[]bSign;
	for (int jj = 0; jj < nSample; jj++) {
		delete[]clsSample[jj];
		delete[]labelPro0[jj];
	}
	delete[]labelPro0;
	delete[]clsSample;

    //开始总运算
	clsSample = iclsSample;
	nSample = nAllSample;

	labelPro0 = new double *[nAllSample];
	for (int i = 0; i < nAllSample; i++)//像素个数
	{
		labelPro0[i] = new double[nLabel];
		memset(labelPro0[i], 0, sizeof(double) * nLabel);
	}
	//普通E-M迭代，相邻两轮误差小于阈值或达到迭代上限，则结束
	double tmpLLLH;
	do
	{
		tmpLLLH = LLLH;
		Count++;
		Expectation();
		Maximization();

		printf("样本数%d 第%d轮：%12.5lf\n", nSample, Count, LLLH);
	} while (fabs(LLLH - tmpLLLH) > erro && Count < 1000);
	
	for (int i = 0; i < nSample; i++) delete[]labelPro0[i];
	delete[]labelPro0;
}

void IEM2::Cluster2(double **iclsSample)
{
	int Count = 0;
	int batch = nAllSample / d;
	int current = 1;
	double tmpLLLH;
	while (current < d) 
	{
		nSample = batch*current ;//当前在训练的像素个数
		labelPro0 = new double *[nSample];
		for (int j = 0; j < nSample; j++)//训练像素个数
		{
			labelPro0[j] = new double[nLabel];
			memset(labelPro0[j], 0, sizeof(double) * nLabel);
		}
		//要初始化clsSample
		clsSample = new double*[nSample];
		for (int ii = 0; ii < nSample; ii++) clsSample[ii] = new double[nDimension];
		
		for (int k = 0; k < batch; k++) {//要选出nSample个点
			int nP;
			nP = (int)(nAllSample / batch * k);//锚点
			for (int g = 0; g < (current); g++) {//是否后延
				int np1 = nP + g;
				int np2 = k * (current) + g;
				for (int kk = 0; kk < nDimension; kk++)
				{
					clsSample[np2][kk] = allSample[np1][kk];
				}
			}
		}
		Expectation();
		Maximization();
		Count++;
		printf("样本数%d 第%d轮：%12.5lf\n", nSample, Count, LLLH *(double)(d));
		do
		{
			tmpLLLH = LLLH;
			Count++;
			Expectation();
			Maximization();
			printf("样本数%d 第%d轮：%12.5lf\n", nSample, Count, LLLH*(double)(d/current));
		} while (fabs(LLLH - tmpLLLH) > (double)erro*(double)(d /current) && Count < 1000);
		current++;
		
		for (int jj = 0; jj < nSample; jj++) {
			delete[]clsSample[jj];
			delete[]labelPro0[jj];
		}
		delete[]labelPro0;
		delete[]clsSample;
	}

	//开始总运算
	clsSample = iclsSample;
	nSample = nAllSample;
	labelPro0 = new double *[nAllSample];
	for (int i = 0; i < nAllSample; i++)//像素个数
	{
		labelPro0[i] = new double[nLabel];
		memset(labelPro0[i], 0, sizeof(double) * nLabel);
	}
	//普通E-M迭代，相邻两轮误差小于阈值或达到迭代上限，则结束
	do
	{
		tmpLLLH = LLLH;
		Count++;
		Expectation();
		Maximization();
		printf("样本数%d 第%d轮：%12.5lf\n", nSample, Count, LLLH);
	} while (fabs(LLLH - tmpLLLH) > erro && Count < 1000);

	for (int i = 0; i < nSample; i++) delete[]labelPro0[i];
	delete[]labelPro0;
}