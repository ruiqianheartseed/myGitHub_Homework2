
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <assert.h>
#include "KMeans.h"
using namespace std;


KMeans::KMeans(int dimNum, int clusterNum)
{
	m_dimNum = dimNum;
	m_clusterNum = clusterNum;

	m_means = new double*[m_clusterNum];
	for (int i = 0; i < m_clusterNum; i++)
	{
		m_means[i] = new double[m_dimNum];
		memset(m_means[i], 0, sizeof(double) * m_dimNum);
	}
	m_maxIterNum = 1000;
	m_endError = 0.0001;
}

KMeans::~KMeans()
{
	for (int i = 0; i < m_clusterNum; i++)
	{
		delete[] m_means[i];
	}
	delete[] m_means;
}


//N 为特征向量数
void KMeans::Cluster(double *data, int N, int *Label)
{
	int size = 0;
	size=N;

	assert(size >= m_clusterNum);

	// Initialize model
	Init(data,N);

	// Recursion
	double* x = new double[m_dimNum];	// Sample data
	int label = -1;		// Class index
	int iterNum = 0;
	double lastCost = 0;
	double currCost = 0;
	int unchanged = 0;
	bool loop = true;
	int* counts = new int[m_clusterNum];
	double** next_means = new double*[m_clusterNum];	// New model for reestimation
	for (int i = 0; i < m_clusterNum; i++)
	{
		next_means[i] = new double[m_dimNum];
	}

	while (loop)
	{
		memset(counts, 0, sizeof(int) * m_clusterNum);
		for (int i = 0; i < m_clusterNum; i++)
		{
			memset(next_means[i], 0, sizeof(double) * m_dimNum);
		}

		lastCost = currCost;
		currCost = 0;

		// Classification
		for (int i = 0; i < size; i++)
		{
			for(int j = 0; j < m_dimNum; j++)
				x[j] = data[i*m_dimNum+j];

			currCost += GetLabel(x, &label);

			counts[label]++;
			for (int d = 0; d < m_dimNum; d++)
			{
				next_means[label][d] += x[d];
			}
		}
		currCost /= size;

		// Reestimation
		for (int i = 0; i < m_clusterNum; i++)
		{
			if (counts[i] > 0)
			{
				for (int d = 0; d < m_dimNum; d++)
				{
					next_means[i][d] /= counts[i];
				}
				memcpy(m_means[i], next_means[i], sizeof(double) * m_dimNum);
			}
		}

		// Terminal conditions
		iterNum++;
        printf("第%d轮：%12.5lf\n", iterNum, currCost);
		if (fabs(lastCost - currCost) < m_endError * lastCost)
		{
			
			unchanged++;
		}
		if (iterNum >= m_maxIterNum || unchanged >= 3)
		{
			loop = false;
		}

		//DEBUG
		//cout << "Iter: " << iterNum << ", Average Cost: " << currCost << endl;
	}

	// Output the label file
	for (int i = 0; i < size; i++)
	{
		for(int j = 0; j < m_dimNum; j++)
				x[j] = data[i*m_dimNum+j];
		GetLabel(x, &label);
		Label[i] = label;
	}
	delete[] counts;
	delete[] x;
	for (int i = 0; i < m_clusterNum; i++)
	{
		delete[] next_means[i];
	}
	delete[] next_means;
}

void KMeans::Init(double *data, int N)
{
	int size = N;
	int inteval = size / m_clusterNum;
	double* sample = new double[m_dimNum];

	// Seed the random-number generator with current time
	srand((unsigned)time(NULL));

	for (int i = 0; i < m_clusterNum; i++)
	{
		int select = inteval * i + (inteval - 1) * rand() / RAND_MAX;
		for (int j = 0; j < m_dimNum; j++)
			sample[j] = data[select*m_dimNum + j];
		memcpy(m_means[i], sample, sizeof(double) * m_dimNum);
	}

	delete[] sample;
	printf("初始聚类中心\n");
	for (int i = 0; i < m_clusterNum; i++)
	{
		printf("pattern %d mean", i);
		for (int j = 0; j < m_dimNum; j++) printf("%12.5lf", m_means[i][j]);
		printf("\n");
	}
	printf("\n");
}


double KMeans::GetLabel(const double* sample, int* label)
{
	double dist = -1;
	for (int i = 0; i < m_clusterNum; i++)
	{
		double temp = CalcDistance(sample, m_means[i], m_dimNum);
		if (temp < dist || dist == -1)
		{
			dist = temp;
			*label = i;
		}
	}
	return dist;
}

double KMeans::CalcDistance(const double* x, const double* u, int dimNum)
{
	double temp = 0;
	for (int d = 0; d < dimNum; d++)
	{
		temp += (x[d] - u[d]) * (x[d] - u[d]);
	}
	return sqrt(temp);
}
