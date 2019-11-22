#include "GMM.h"
#include "memory.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#define erro 0.1 //�������
#define randInitial 0
#define PI 3.1415926

//һ����С�������������Э�����������ʽΪ0�Լ���˹�ܶȷֲ�����Ϊ0������
#define CONST_E 0.000000000001

GMM::GMM(int tnSample, int tnDimension, int tnLabel)
{
	nSample = tnSample;
	nDimension = tnDimension;
	nLabel = tnLabel;

	//labelPro[i][j]��ʾ��i���������ڵ�j��ĸ���,��ʼ��Ϊ0
	labelPro = new double *[nSample];
	for (int i = 0; i < nSample; i++)
	{
		labelPro[i] = new double[nLabel];
		memset(labelPro[i], 0, sizeof(double) * nLabel);
	}

	//labelU[i]��ʾ��i��ľ�ֵ���������ʼ����ֵͨ�������k������means�ķ����õ�
	labelU = new double *[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelU[i] = new double[nDimension];
	}

	//labelCOV[i]��ʾ��i���Э�������,����Э��������ʼ���ɵ�λ��
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

	//labelPripro[i]��ʾ��i����������,��ʼ��Ϊ1 / _nPat
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
	//����ԭ����
	double **tmpMat = new double *[nDimension];
	for (int i = 0; i < nDimension; i++)
	{
		tmpMat[i] = new double[nDimension];
		memcpy(tmpMat[i], iMat[i], sizeof(double) * nDimension);
	}

	//����һ����λ��
	double **tmpI = new double *[nDimension];
	for (int i = 0; i < nDimension; i++)
	{
		tmpI[i] = new double[nDimension];
		memset(tmpI[i], 0, sizeof(double) * nDimension);
		tmpI[i][i] = 1;
	}

	//Gaussian��Ԫ���������--������Ԫ
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

	//Gaussian��Ԫ���������--������Ԫ
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

	//������ʱ�ռ�
	for (int i = 0; i < nDimension; i++) delete[]tmpMat[i];
	delete[]tmpMat;

	return tmpI;
}

double GMM::Determinant(double **iMat)
{
	//����ԭ����
	double **tmpMat = new double *[nDimension];
	for (int i = 0; i < nDimension; i++)
	{
		tmpMat[i] = new double[nDimension];
		memcpy(tmpMat[i], iMat[i], sizeof(double) * nDimension);
	}

	//Gaussian��Ԫ��--�õ��Խ���ʽ
	for (int i = 0; i < nDimension; i++)
	{
		for (int j = i + 1; j < nDimension; j++)
		{
			double lfTmp = -tmpMat[j][i] / tmpMat[i][i];
			for (int k = 0; k < nDimension; k++) tmpMat[j][k] += lfTmp * tmpMat[i][k];
		}
	}

	//����Խ��߳˻�
	double det = 1;
	for (int i = 0; i < nDimension; i++) det *= tmpMat[i][i];

	//������ʱ�ռ�
	for (int i = 0; i < nDimension; i++) delete[]tmpMat[i];
	delete[]tmpMat;

	return det;
}

double GMM::Gaussian(double *clsSample, double *labelU, double **labelCOV)
{
	//�������������ά��˹�ֲ��ܶȺ���

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
		double Sum = 0;//���ڼ���likelihood
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

		//��󻯲���--��_pplfU[j]
		memset(labelU[j], 0, sizeof(double) * nDimension);
		for (int i = 0; i < nSample; i++)
			for (int k = 0; k < nDimension; k++)
				labelU[j][k] += labelPro[i][j] * clsSample[i][k];
		for (int k = 0; k < nDimension; k++) labelU[j][k] /= Tmp1;

		//��󻯲���--��_ppplfDelta[j]
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

		//��󻯲���--��_plfPi[j]
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
		for (int i = 0; i < nLabel; i++)//����ÿһ�࣬���rollһ����
		{
			int nPt;
			do {
				nPt = (int)(((double)rand() / (double)RAND_MAX) * (nSample - 0.1));
			} while (bSign[nPt]);//���roll�ĵ��ظ��˾���rollһ��ֱ�����ظ�
			bSign[nPt] = true;//roll���ͱ���ѷ���
			memcpy(labelU[i], clsSample[nPt], sizeof(double) * nDimension);
		}
	}
	else
	{
		for (int i = 0; i < nLabel; i++)
		{
			int nPt;
			nPt = (int)(nSample / nLabel * i + nSample / (2 * nLabel));//�̶���ʼ��,����ȡ��
			bSign[nPt] = true;
			memcpy(labelU[i], clsSample[nPt], sizeof(double) * nDimension);
		}
	}
	delete[]bSign;


	printf("��ʼ��������\n");
	for (int i = 0; i < nLabel; i++)
	{
		printf("pattern %d mean", i);
		for (int j = 0; j < nDimension; j++) printf("%12.5lf", labelU[i][j]);
		printf("\n");
	}
	printf("\n");

	//E-M�����������������С����ֵ��ﵽ�������ޣ������
	Expectation();
	Maximization();

	int Count = 1;
	printf("��%d�֣�%12.5lf\n", Count, LLLH);
	double tmpLLLH;
	do
	{
		tmpLLLH = LLLH;
		Count++;
		Expectation();
		Maximization();

		printf("��%d�֣�%12.5lf\n", Count, LLLH);
	} while (fabs(LLLH - tmpLLLH) > erro && Count < 1000 );

}

