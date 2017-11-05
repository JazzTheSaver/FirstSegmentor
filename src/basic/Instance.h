#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void clear()
	{
		m_chars.clear();
		m_bichars_l.clear();
		m_bichars_r.clear();
		m_labels.clear();
	}
	void toVector(const vector<string>& labels, vector<vector<int>>& vec) const{
		int size = labels.size();
		int i = 0, j;
		vector<int> temp;
		while (i < size) {
			temp.clear();
			for (j = i + 1; j < size; j++) {
				if (labels[j] == "S") {
					temp.push_back(i);
					temp.push_back(j);
					//cout << temp[0]; cout << temp[1] << endl;
					vec.push_back(temp);
					i = j;
					break;
				}
			}
			if (j == size) {
				temp.push_back(i);
				temp.push_back(j);
				//cout << temp[0]; cout << temp[1] << endl;
				vec.push_back(temp);
				i = j;
			}

		}
	}

	std::vector<vector<int>> findSame(const std::vector<vector<int>> &nLeft, const std::vector<vector<int>> &nRight) {
		std::vector<vector<int>> nResult;
		for (std::vector<vector<int>>::const_iterator nIterator = nLeft.begin(); nIterator != nLeft.end(); nIterator++)
		{
			if (std::find(nRight.begin(), nRight.end(), *nIterator) != nRight.end())
				nResult.push_back(*nIterator);
		}
		return nResult;
	}
	void evaluate(const vector<string>& predict_labels , Metric& eval) const{
		vector<vector<int>> gold;
		vector<vector<int>> predict;
		int pre_size = predict_labels.size();
		vector<int> temp;
	/*	cout << endl;
		for (int i = 0; i < m_labels.size(); i++) {
			cout << m_labels[i]; cout << predict_labels[i]<<endl;
		}*/
	
		toVector(predict_labels, predict);

		toVector(m_labels, gold);

		vector<vector<int>> nResult;
		for (vector<vector<int>>::const_iterator nIterator = gold.begin(); nIterator != gold.end(); nIterator++)
		{
			if (std::find(predict.begin(), predict.end(), *nIterator) != predict.end())
				nResult.push_back(*nIterator);
		}
		eval.correct_label_count += nResult.size();
		eval.predicated_label_count += predict.size();
		eval.overall_label_count += gold.size();
	}
	//void evaluate(const string& predict_label, Metric& eval) const
	//{
	//	//if (predict_label == m_labels)
	//	//	eval.correct_label_count++;
	//	//eval.overall_label_count++;
	//}

	void copyValuesFrom(const Instance& anInstance)
	{
		allocate(anInstance.size());
		m_labels = anInstance.m_labels;
		m_chars = anInstance.m_chars;
		m_bichars_l = anInstance.m_bichars_l;
		m_bichars_r = anInstance.m_bichars_r;
	}

	void assignLabel(const vector<string>& resulted_label) {
		m_labels = resulted_label;
	}

	int size() const {
		return m_chars.size();
	}
	void allocate(int length)
	{
		clear();
		m_chars.resize(length);
		m_bichars_l.resize(length);
		m_bichars_r.resize(length);
	}
public:
	vector<string> m_chars;
	vector<string> m_bichars_l;
	vector<string> m_bichars_r;
	vector<string> m_labels;  //character's labels
	
};

#endif /*_INSTANCE_H_*/
