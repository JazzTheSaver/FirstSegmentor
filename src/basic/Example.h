#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;


class Example
{
public:
	vector<string> m_chars;
	vector<string> m_bichars_l;
	vector<string> m_bichars_r;
	vector<vector<dtype>> m_labels;

public:
	void clear()
	{
		m_chars.clear();
		m_bichars_l.clear();
		m_bichars_r.clear();
		m_labels.clear();
	}
};

#endif /*_EXAMPLE_H_*/