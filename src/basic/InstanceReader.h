#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include <sstream>
#include "Utf.h"
using namespace std;
/*
this class reads conll-format data (10 columns, no srl-info)
*/
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {

			m_instance.clear();
		string strLine;
		if (!my_getline(m_inf, strLine)) {
			return NULL;
		}
		
		getCharactersFromString(strLine, m_instance.m_chars);
	
		vector<string>::iterator it;
		int j = 0;
		for (int i = 0; i < m_instance.m_chars.size(); i++) {
			if (i == 0) {
				m_instance.m_labels.push_back("S");
				j++;
			}
			else {
				if (m_instance.m_chars[i] != " ") {
					if (m_instance.m_chars[i - 1] != " ") {
						m_instance.m_labels.push_back("A");
					}
					else {
						m_instance.m_labels.push_back("S");
					}
					j++;
				}
			}
		}

		for (it = m_instance.m_chars.begin(); it != m_instance.m_chars.end();it++) {
			if (*it == " ") {
				it = m_instance.m_chars.erase(it);
			}
		}
		//ofstream outfile;
		//string ofile = "D:\\test.txt";
		//outfile.open(ofile.c_str());

		int chars_size = m_instance.m_chars.size();
		vector<string> temp_vec;
		for (int i = 0; i < chars_size; i++) {
			temp_vec.push_back(m_instance.m_chars[i]);
		}
		
	
		string temp1;
		temp1 += "<s>";
		temp1 += temp_vec[0];
		m_instance.m_bichars_l.push_back(temp1);
		for (int i = 0; i < temp_vec.size() - 1; i++) {
			temp1.clear();
			temp1 += temp_vec[i];
			temp1 += temp_vec[i + 1];
			m_instance.m_bichars_l.push_back(temp1);   //initial bichar
			m_instance.m_bichars_r.push_back(temp1);   //initial bichar
		}
		string temp2 = temp_vec[temp_vec.size() - 1];
		temp2 += "<e>";
		m_instance.m_bichars_r.push_back(temp2);   //initial bichar

		return &m_instance;

	

	}

};

#endif




