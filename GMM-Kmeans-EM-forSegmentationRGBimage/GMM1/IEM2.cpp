#include "IEM2.h"
#include "memory.h"
#include "stdlib.h"
#include "stdio.h"
#include "math.h"
#include "time.h"
#define d 10 //������
#define nd 200 //Ԥ��ѵ��
#define erro 0.1 //�������
#define randInitial 0
#define PI 3.1415926
#define CONST_E 0.000000000001//һ����С�������������Э�����������ʽΪ0�Լ���˹�ܶȷֲ�����Ϊ0������
//IEM2�ĺ����ǣ����²���ֻʹ��ÿ��һ���ֵ����أ����������ҪԤ���������صı�ǩ��

//IEM2����������ṹ�����Լ���ʼ��
IEM2::IEM2(int tnSample, int tnDimension, int tnLabel, double **iclsSample)
{
	allSample = iclsSample;//�������������������ֻ���labelPro
	nAllSample = tnSample;//ȫ�����ظ���
	nDimension = tnDimension;
	nLabel = tnLabel;

	//labelPro[i][j]��ʾ��i���������ڵ�j��ĸ���,��ʼ��Ϊ0���������
	labelPro = new double *[nAllSample];
	for (int i = 0; i < nAllSample; i++)//���ظ���
	{
		labelPro[i] = new double[nLabel];
		memset(labelPro[i], 0, sizeof(double) * nLabel);
	}

	//labelU[i]��ʾ��i��ľ�ֵ���������ʼ����ֵͨ�������k������means�ķ����õ������߹̶��õ�
	labelU = new double *[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelU[i] = new double[nDimension];
	}
	srand((unsigned)time(NULL));
	rand();

	//labelU[nLabel]�ĳ�ʼ��
	bool *bSign = new bool[nAllSample];
	memset(bSign, false, sizeof(bool) * nAllSample);//ÿһ��������false
	if (randInitial == 1)
	{
		for (int i = 0; i < nLabel; i++)//����ÿһ�࣬���rollһ����
		{
			int nPt;
			do {
				nPt = (int)(((double)rand() / (double)RAND_MAX) * (nAllSample - 0.1));
			} while (bSign[nPt]);//���roll�ĵ��ظ��˾���rollһ��ֱ�����ظ�
			bSign[nPt] = true;//roll���ͱ���ѷ���
			memcpy(labelU[i], allSample[nPt], sizeof(double) * nDimension);
		}
	}
	else
	{
		for (int i = 0; i < nLabel; i++)
		{
			int nPt;
			nPt = (int)(nAllSample / nLabel * i + nAllSample / (2 * nLabel));//�̶���ʼ��,����ȡ��
			bSign[nPt] = true;
			memcpy(labelU[i], allSample[nPt], sizeof(double) * nDimension);
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

	//labelCOV[i]��ʾ��i���Э�������,����Э��������ʼ���ɵ�λ��
	labelCOV = new double **[nLabel];
	for (int i = 0; i < nLabel; i++)
	{
		labelCOV[i] = new double *[nDimension];
		for (int j = 0; j < nDimension; j++)
		{
			labelCOV[i][j] = new double[nDimension];//i�࣬ÿһ�඼��3*3�ľ���һ��һ�г�ʼ��
			memset(labelCOV[i][j], 0, sizeof(double) * nDimension);//�ȰѶ�Ӧ��ȫ����Ϊ0
			labelCOV[i][j][j] = 1;//Ȼ���ٰѶԽ�����1
		}
	}

	//labelPripro[i]��ʾ��i����������,��ʼ��Ϊ1 / _nPat
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

//3*3�����˹��Ԫ������
double **IEM2::Inverse(double **iMat)
{
	//����ԭ����tmpMat
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
//��3*3����ʽ��ֵ
double IEM2::Determinant(double **iMat)
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
//���ά��˹�ֲ��ܶȺ�������ǰ���������/����/Э�������
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
		double Sum = 0;//���ڼ���likelihood
		for (int j = 0; j < nLabel; j++)
		{
			double Tmp = 0;
			for (int l = 0; l < nLabel; l++)
			{
				double G = Gaussian(clsSample[i], labelU[l], labelCOV[l]);//��˹�ֲ��ܶ�
				Tmp += labelPripro[l] * G;
			}
			double G = Gaussian(clsSample[i], labelU[j], labelCOV[j]);//�����˹�ֲ��ܶ�
			labelPro0[i][j] = labelPripro[j] * G / Tmp;
			if (nSample == nAllSample) labelPro[i][j] = labelPro0[i][j];//�����labelPro�ǻᱻ���������,��������
			Sum += labelPripro[j] * G;//����Ӱ���б��ֻ��sum
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

		//��󻯲���--��_pplfU[j]
		memset(labelU[j], 0, sizeof(double) * nDimension);//�ְ�j��������0�ˣ�

		for (int i = 0; i < nSample; i++)
			for (int k = 0; k < nDimension; k++)
				labelU[j][k] += labelPro0[i][j] * clsSample[i][k];
		for (int k = 0; k < nDimension; k++) labelU[j][k] /= Tmp1;

		//��󻯲���--��_ppplfDelta[j]
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

		//��󻯲���--��_plfPi[j]
		labelPripro[j] = Tmp1 / nSample;
	}
}

void IEM2::Cluster(double **iclsSample)
{
	int Count = 0;
	int batch = nAllSample / d;
	nSample = batch;//��ǰ��ѵ�������ظ���
	//labelPro0[i][j]��ʾѵ��ʱi���������ڵ�j��ĸ���
	labelPro0 = new double *[nSample];
	for (int j = 0; j < nSample; j++)//ѵ�����ظ���
	{
		labelPro0[j] = new double[nLabel];
		memset(labelPro0[j], 0, sizeof(double) * nLabel);
	}
	//Ҫ��ʼ��clsSample
	clsSample = new double*[nSample];
	for (int ii = 0; ii < nSample; ii++)
	{
		clsSample[ii] = new double[nDimension];
	}
	srand((unsigned)time(NULL));
	rand();
	bool *bSign = new bool[nAllSample];

	//��ǰnd�����������ǲ����ӵ�
	for (int i = 0; i < nd; i++) {
		memset(bSign, false, sizeof(bool) * nAllSample);//ÿһ��������false
		for (int k = 0; k < batch; k++) {
			int nPt;
			do {
				nPt = (int)(((double)rand() / (double)RAND_MAX) * (nAllSample - 0.1));
			} while (bSign[nPt]);//���roll�ĵ��ظ��˾���rollһ��ֱ�����ظ�
			bSign[nPt] = true;//roll���ͱ���ѷ���
			for (int kk = 0; kk < nDimension; kk++)
			{
				clsSample[k][kk] = allSample[nPt][kk];
			}
		}		
		Expectation();
		Maximization();
		Count++;
		printf("������%d ��%d�֣�%12.5lf\n", nSample, Count, LLLH *(double)(d));
	}

	delete[]bSign;
	for (int jj = 0; jj < nSample; jj++) {
		delete[]clsSample[jj];
		delete[]labelPro0[jj];
	}
	delete[]labelPro0;
	delete[]clsSample;

    //��ʼ������
	clsSample = iclsSample;
	nSample = nAllSample;

	labelPro0 = new double *[nAllSample];
	for (int i = 0; i < nAllSample; i++)//���ظ���
	{
		labelPro0[i] = new double[nLabel];
		memset(labelPro0[i], 0, sizeof(double) * nLabel);
	}
	//��ͨE-M�����������������С����ֵ��ﵽ�������ޣ������
	double tmpLLLH;
	do
	{
		tmpLLLH = LLLH;
		Count++;
		Expectation();
		Maximization();

		printf("������%d ��%d�֣�%12.5lf\n", nSample, Count, LLLH);
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
		nSample = batch*current ;//��ǰ��ѵ�������ظ���
		labelPro0 = new double *[nSample];
		for (int j = 0; j < nSample; j++)//ѵ�����ظ���
		{
			labelPro0[j] = new double[nLabel];
			memset(labelPro0[j], 0, sizeof(double) * nLabel);
		}
		//Ҫ��ʼ��clsSample
		clsSample = new double*[nSample];
		for (int ii = 0; ii < nSample; ii++) clsSample[ii] = new double[nDimension];
		
		for (int k = 0; k < batch; k++) {//Ҫѡ��nSample����
			int nP;
			nP = (int)(nAllSample / batch * k);//ê��
			for (int g = 0; g < (current); g++) {//�Ƿ����
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
		printf("������%d ��%d�֣�%12.5lf\n", nSample, Count, LLLH *(double)(d));
		do
		{
			tmpLLLH = LLLH;
			Count++;
			Expectation();
			Maximization();
			printf("������%d ��%d�֣�%12.5lf\n", nSample, Count, LLLH*(double)(d/current));
		} while (fabs(LLLH - tmpLLLH) > (double)erro*(double)(d /current) && Count < 1000);
		current++;
		
		for (int jj = 0; jj < nSample; jj++) {
			delete[]clsSample[jj];
			delete[]labelPro0[jj];
		}
		delete[]labelPro0;
		delete[]clsSample;
	}

	//��ʼ������
	clsSample = iclsSample;
	nSample = nAllSample;
	labelPro0 = new double *[nAllSample];
	for (int i = 0; i < nAllSample; i++)//���ظ���
	{
		labelPro0[i] = new double[nLabel];
		memset(labelPro0[i], 0, sizeof(double) * nLabel);
	}
	//��ͨE-M�����������������С����ֵ��ﵽ�������ޣ������
	do
	{
		tmpLLLH = LLLH;
		Count++;
		Expectation();
		Maximization();
		printf("������%d ��%d�֣�%12.5lf\n", nSample, Count, LLLH);
	} while (fabs(LLLH - tmpLLLH) > erro && Count < 1000);

	for (int i = 0; i < nSample; i++) delete[]labelPro0[i];
	delete[]labelPro0;
}