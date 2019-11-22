
#ifndef IEM2_H
#define IEM2_H

using namespace std;

class IEM2
{
private:
	int nSample;     //ÿ�ִ��������������
	int nDimension;     //������ά��
	int nLabel;     //�����
	int nAllSample;    //���д��������������
	double **clsSample;   //�����������
	double **labelU;        //����ÿһ������ľ�ֵ����
	double *labelPripro;         //����ÿһ��������������
	double ***labelCOV;  //����ÿһ�������Э�������
	double **allSample;  //���е�����
	double LLLH;            //�洢log likelihood����ֵ,��ֵ��E-Step����м���
	//����Ŀ�꼴Ҫʹ��ֵ���
	double **labelPro0;//��������е�i���������ڵ�j��ĸ���
public:
	double **labelPro;        //��i���������ڵ�j��ĸ���
	/**
	���캯���������ڴ��ʼ��
	param nVec �������������Ŀ
	param nDim �����������ά��
	param nPat ����Ϊ������
	*/
	IEM2(int tnSample, int tnDimension, int tnLabel, double **iclsSample);

	/**
	�����������ͷ�����������ڴ�
	*/
	virtual ~IEM2();

	/**
	����
	param pplfVector ��������������������ά���Լ������ڹ��캯����ָ��
	*/
	void Cluster(double **clsSample);
	void Cluster2(double **clsSample);
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

