#include "GMM.h"
#include "memory.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#define erro 0.1 //允许误差
#define randInitial 0
#define PI 3.1415926

//一个很小的数，用来解决协方差矩阵行列式为0以及高斯密度分布函数为0的问题
#define CONST_E 0.000000000001

GMM::GMM(int tnSample, int tnDimension, int tnLabel)
{
	nSample = tnSample;
	nDimension = tnDimension;
	nLabel = tnLabel;

	//labelPro[i][j]表示第i个向量属于第j类的概率,初始化为0
	labelPro = new double *[nSample];
	for (int i = 0; i < nSample; i++)
	{
		labelPro[i] = new double[nLabel];
		memset(labelPro[i], 0, sizeof(double) * nLabel);
	}

	//labelU[i]表示第i类的均值向量，其初始化的值通过随机到k个中心means的方法得到
	labelU = new double *[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelU[i] = new double[nDimension];
	}

	//labelCOV[i]表示第i类的协方差矩阵,并把协方差矩阵初始化成单位阵
	labelCOV = new double **[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelCOV[i] = new double *[nDimension];
		for (int j = 0; j < nDimension; j++)
		{
			labelCOV[i][j] = new double[nDimension];
			memset(labelCOV[i][j], 0, sizeof(double) * nDimension);
			labelCOV[i][j][j] = 1;
		}
	}

	//labelPripro[i]表示第i类的先验概率,初始化为1 / _nPat
	labelPripro = new double[nLabel];
	for (int i = 0; i < nLabel; i++) labelPripro[i] = 1.0 / nLabel;
}

GMM::~GMM()
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

double **GMM::Inverse(double **iMat)
{
	//复制原矩阵
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

double GMM::Determinant(double **iMat)
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

double GMM::Gaussian(double *clsSample, double *labelU, double **labelCOV)
{
	//整个函数是求多维高斯分布密度函数

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

void GMM::Expectation()
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
				double G = Gaussian(clsSample[i], labelU[l], labelCOV[l]);
				Tmp += labelPripro[l] * G;
			}
			double G = Gaussian(clsSample[i], labelU[j], labelCOV[j]);
			labelPro[i][j] = labelPripro[j] * G / Tmp;
			Sum += labelPripro[j] * G;
		}
		LLLH += log(Sum);
	}
}

void GMM::Maximization()
{
	//M-Step
	for (int j = 0; j < nLabel; j++)
	{
		double Tmp1 = 0;
		for (int i = 0; i < nSample; i++) Tmp1 += labelPro[i][j];

		//最大化步骤--求_pplfU[j]
		memset(labelU[j], 0, sizeof(double) * nDimension);
		for (int i = 0; i < nSample; i++)
			for (int k = 0; k < nDimension; k++)
				labelU[j][k] += labelPro[i][j] * clsSample[i][k];
		for (int k = 0; k < nDimension; k++) labelU[j][k] /= Tmp1;

		//最大化步骤--求_ppplfDelta[j]
		for (int i = 0; i < nDimension; i++)
			memset(labelCOV[j][i], 0, sizeof(double) * nDimension);
		for (int i = 0; i < nSample; i++)
		{
			double *Tmp2 = new double[nDimension];
			for (int a = 0; a < nDimension; a++) Tmp2[a] = clsSample[i][a] - labelU[j][a];
			for (int a = 0; a < nDimension; a++)
				//for (int b = 0; b <= _nDim; b++)
				labelCOV[j][a][a] += labelPro[i][j] * Tmp2[a] * Tmp2[a];
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

void GMM::Cluster(double **iclsSample)
{
	clsSample = iclsSample;

	srand((unsigned)time(NULL));
	rand();

	bool *bSign = new bool[nSample];
	memset(bSign, false, sizeof(bool) * nSample);
	if (randInitial == 1)
	{
		for (int i = 0; i < nLabel; i++)//对于每一类，随机roll一个点
		{
			int nPt;
			do {
				nPt = (int)(((double)rand() / (double)RAND_MAX) * (nSample - 0.1));
			} while (bSign[nPt]);//如果roll的点重复了就再roll一次直到不重复
			bSign[nPt] = true;//roll过就标记已访问
			memcpy(labelU[i], clsSample[nPt], sizeof(double) * nDimension);
		}
	}
	else
	{
		for (int i = 0; i < nLabel; i++)
		{
			int nPt;
			nPt = (int)(nSample / nLabel * i + nSample / (2 * nLabel));//固定初始化,均匀取点
			bSign[nPt] = true;
			memcpy(labelU[i], clsSample[nPt], sizeof(double) * nDimension);
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

	//E-M迭代，相邻两轮误差小于阈值或达到迭代上限，则结束
	Expectation();
	Maximization();

	int Count = 1;
	printf("第%d轮：%12.5lf\n", Count, LLLH);
	double tmpLLLH;
	do
	{
		tmpLLLH = LLLH;
		Count++;
		Expectation();
		Maximization();

		printf("第%d轮：%12.5lf\n", Count, LLLH);
	} while (fabs(LLLH - tmpLLLH) > erro && Count < 1000 );

}

