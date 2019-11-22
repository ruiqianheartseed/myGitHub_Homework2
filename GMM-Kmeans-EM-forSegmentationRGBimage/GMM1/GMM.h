
#ifndef GMM_H
#define GMM_H

using namespace std;

class GMM
{
private:
	int nSample;     //���������������
	int nDimension;     //������ά��
	int nLabel;     //�����

	double **clsSample;   //�����������
	double **labelU;        //����ÿһ������ľ�ֵ����
	double *labelPripro;         //����ÿһ��������������
	double ***labelCOV;  //����ÿһ�������Э�������

	double LLLH;            //�洢log likelihood����ֵ,��ֵ��E-Step����м���
	//����Ŀ�꼴Ҫʹ��ֵ���
public:
	double **labelPro;        //��i���������ڵ�j��ĸ���
	/**
	���캯���������ڴ��ʼ��
	param nVec �������������Ŀ
	param nDim �����������ά��
	param nPat ����Ϊ������
	*/
	GMM(int tnSample, int tnDimension, int tnLabel);

	/**
	�����������ͷ�����������ڴ�
	*/
	virtual ~GMM();

	/**
	����
	param pplfVector ��������������������ά���Լ������ڹ��캯����ָ��
	*/
	void Cluster(double **clsSample);

private:
	/**
	E-Step
	*/
	void Expectation();

	/**
	M-Step
	*/
	void Maximization();

	/**
	����������ʽ
	param pplfMat ��������ʽ�ľ���
	����ֵ ����ʽ��ֵ
	*/
	double Determinant(double **iMat);

	//��������
	double ** Inverse(double **iMat);

	/**
	��ά��˹�ֲ��ܶȺ���
	param plfVec ��ά�ռ��еĵ����꣬��Ҫ��õ��ϵĸ����ܶ�
	param plfU ��˹�ֲ��ľ�ֵ����
	param pplfDelta ��˹�ֲ���Э�������
	����ֵ ��ά�ռ��ж�Ӧ��ĸ����ܶ�
	*/
	double Gaussian(double *clsSample, double *labelU, double **labelCOV);
};


#endif
