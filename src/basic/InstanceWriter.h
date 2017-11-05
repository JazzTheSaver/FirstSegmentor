#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include <sstream>

using namespace std;

class InstanceWriter : public Writer
{
public:
	InstanceWriter(){}
	~InstanceWriter(){}
	int write(const Instance *pInstance)
	{
	  if (!m_outf.is_open()) return -1;
		

	  //const string &label = pInstance->m_label;

	  //m_outf << label << "\t";
	  //vector<string> words = pInstance->m_words;
	  //int word_size = words.size();
	  //for (int idx = 0; idx < word_size; idx++)
		 // m_outf << words[idx] << " ";
	  //m_outf << endl;
	  //return 0;





	const vector<string> &labels = pInstance->m_labels;
	int labels_size = labels.size();
	vector<string> chars = pInstance->m_chars;
	int char_size = chars.size();
	//for (int idx = 0; idx < char_size; idx++)
	//	m_outf << chars[idx] << " ";
	//m_outf << endl;

	for (int i = 0; i < labels_size; i++) {
		if (i == 0) {
			m_outf << chars[i];
		}
		else {
			if (labels[i] == "S") {
				m_outf << "\t" << chars[i];
			}else {
				m_outf << chars[i];

			}
		}
	}
	m_outf << endl;
	return  0;
	  
	  return  0;





	}
};

#endif

