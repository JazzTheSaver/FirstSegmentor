#include "BiLSTMSegmentor.h"

#include <chrono>
#include "Argument_helper.h"

Classifier::Classifier(){
	// TODO Auto-generated constructor stub
	srand(0);
}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
}

int Classifier::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0){
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;
	int numInstance;
	m_driver._modelparams.labelAlpha.clear();
	m_driver._modelparams.charsAlpha.clear();
	m_driver._modelparams.bicharsAlpha.clear();
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		const vector<string> &chars = pInstance->m_chars;
		const vector<string> &bichars_l = pInstance->m_bichars_l;
		const vector<string> &bichars_r = pInstance->m_bichars_r;
		const vector<string> &labels = pInstance->m_labels;



		int chars_num = chars.size();
		for (int i = 0; i < chars_num; i++) {
			m_char_stats[chars[i]]++;
		}

		m_driver._modelparams.charsAlpha.initial(m_char_stats,m_options.charCutOff);

		int bichars_num = bichars_l.size();
		int b1ichars_num = bichars_r.size();
		for (int i = 0; i < bichars_num; i++) {
			m_bichar_stats[bichars_l[i]]++;
		}
		m_bichar_stats[bichars_r[bichars_num-1]]++;
		m_driver._modelparams.bicharsAlpha.initial(m_bichar_stats, m_options.bicharCutOff);
		
		for (int i = 0; i < labels.size(); i++) {
			m_driver._modelparams.labelAlpha.from_string(labels[i]);
		}
		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;
	cout << "Chars num: " << m_driver._modelparams.charsAlpha.size() << endl;
	cout << "BiChars num: " << m_driver._modelparams.bicharsAlpha.size() << endl;
	cout << "Labels num: " << m_driver._modelparams.labelAlpha.size() << endl;
	/*int count = 0;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		const vector<string> &chars = pInstance->m_chars;
		for (int i = 0; i < chars.size(); i++) {
			string ch = chars[i];
			int num = m_driver._modelparams.charsAlpha.from_string(ch);
			if (num != (-1)) {
				count++;
			}
		}
	}*/

	m_driver._modelparams.charsAlpha.set_fixed_flag(true);
	m_driver._modelparams.bicharsAlpha.set_fixed_flag(true);
	//ofstream outfile;
	//string ofile = "d:\\test.txt";
	//outfile.open(ofile.c_str(),ios::app);
	//outfile << m_driver._modelparams.charsAlpha.from_id;

	//cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;
	//cout << "Sparse Feature num: " << m_feat_stats.size() << endl;
	//cout << "Word num: " << m_word_stats.size() << endl;


	return 0;
}

int Classifier::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;

	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		
		const vector<string> &chars = pInstance->m_chars;
		int chars_num = chars.size();
		for (int i = 0; i < chars_num; i++) {
			m_char_stats[chars[i]]++;
		}

		m_driver._modelparams.charsAlpha.initial(m_char_stats);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;

	return 0;
}

//
//void Classifier::extractFeature(Feature& feat, const Instance* pInstance) {
//	//feat.clear();
//	//feat.m_words = pInstance->m_words;
//	//feat.m_sparse_feats = pInstance->m_sparse_feats;
//}

void Classifier::convert2Example(const Instance* pInstance, Example& exam) {
	exam.clear();
	//const string &orcale = pInstance->m_label;
	const vector<string> &oracle = pInstance->m_labels;

	int numLabel = oracle.size();
	vector<vector<dtype>> curlabels;
	vector<dtype> temp;

	
	string str = "S";

	for (int j = 0; j < numLabel; ++j) {
		temp.clear();
		string s = oracle[j];
		if (str.compare(s) == 0) {
			temp.push_back(1.0);
			temp.push_back(0.0);
		}else {
			temp.push_back(0.0);
			temp.push_back(1.0);
		}
		curlabels.push_back(temp);
	}
	exam.m_chars = pInstance->m_chars;
	exam.m_bichars_l = pInstance->m_bichars_l;
	exam.m_bichars_r = pInstance->m_bichars_r;
	exam.m_labels = curlabels;
}

void Classifier::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam);
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;
}

void Classifier::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;
	m_pipe.readInstances(trainFile, trainInsts);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts);

	//Ensure that each file in m_options.testFiles exists!
	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	cout << trainInsts.size();
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
	}

	createAlphabet(trainInsts);
	addTestAlpha(devInsts);
	addTestAlpha(testInsts);
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		addTestAlpha(otherInsts[idx]);
	}

	vector<Example> trainExamples, devExamples, testExamples;

	initialExamples(trainInsts, trainExamples);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);


	vector<int> otherInstNums(otherInsts.size());
	vector<vector<Example> > otherExamples(otherInsts.size());
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		initialExamples(otherInsts[idx], otherExamples[idx]);
		otherInstNums[idx] = otherExamples[idx].size();
	}

	m_char_stats[unknownkey] = m_options.charCutOff + 1;
 	m_driver._modelparams.charsAlpha.initial(m_char_stats, m_options.charCutOff);
	m_bichar_stats[unknownkey] = m_options.bicharCutOff + 1;
	m_driver._modelparams.bicharsAlpha.initial(m_bichar_stats, m_options.charCutOff);
	if (m_options.charFile != "") {
		m_driver._modelparams.chars.initial(&m_driver._modelparams.charsAlpha, m_options.charFile, m_options.charEmbFineTune);
	}
	else{
		m_driver._modelparams.chars.initial(&m_driver._modelparams.charsAlpha, m_options.charEmbSize, m_options.charEmbFineTune);
	}
	if (m_options.bicharFile != "") {
		m_driver._modelparams.l_bichars.initial(&m_driver._modelparams.bicharsAlpha, m_options.bicharFile, m_options.bicharEmbFineTune);
	}
	else {
		m_driver._modelparams.l_bichars.initial(&m_driver._modelparams.bicharsAlpha, m_options.bicharEmbSize, m_options.bicharEmbFineTune);
	}
	if (m_options.bicharFile != "") {
		m_driver._modelparams.r_bichars.initial(&m_driver._modelparams.bicharsAlpha, m_options.bicharFile, m_options.bicharEmbFineTune);
	}
	else {
		m_driver._modelparams.r_bichars.initial(&m_driver._modelparams.bicharsAlpha, m_options.bicharEmbSize, m_options.bicharEmbFineTune);
	}

	m_driver._hyperparams.setRequared(m_options);
//	cout<<m_driver._modelparams.bichars.E.val.mat();
	m_driver.initial();


	dtype bestDIS = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		auto time_start = std::chrono::high_resolution_clock::now();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_driver.train(subExamples, curUpdateIter);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
			//	m_driver.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_driver.updateModel();

		}
		auto time_end = std::chrono::high_resolution_clock::now();
		std::cout << "Train finished. Total time taken is: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl;


		if (devNum > 0) {
			auto time_start = std::chrono::high_resolution_clock::now();
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			for (int idx = 0; idx < devExamples.size(); idx++) {
				vector<string> result_label;
///////
				predict(devExamples[idx], result_label);
				 

				for (int i = 0; i < result_label.size(); i++) {
//					cout << result_label[i];
				}
				devInsts[idx].evaluate(result_label, metric_dev);


				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_label);
					decodeInstResults.push_back(curDecodeInst);
				}
			}

			auto time_end = std::chrono::high_resolution_clock::now();
			std::cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl;
			std::cout << "dev:" << std::endl;
//todo
			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				auto time_start = std::chrono::high_resolution_clock::now();
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					vector<string> result_label;
					predict(testExamples[idx], result_label);

					testInsts[idx].evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}

				auto time_end = std::chrono::high_resolution_clock::now();
				std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(time_end - time_start).count() << "s" << std::endl;
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherExamples.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
					vector<string> result_label;
					predict(otherExamples[idx][idy], result_label);

					otherInsts[idx][idy].evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Classifier::predict(const Example& example, vector<string>& output) {
	//assert(features.size() == words.size());

	vector<int> labelIdx;
	labelIdx.resize(example.m_labels.size());

	m_driver.predict(example, labelIdx);
	
	int labelIdx_size = labelIdx.size();
	//cout << m_driver._modelparams.charsAlpha.from_id(labelIdx);
	for (int i = 0; i < labelIdx_size;i++) {
		output.push_back(m_driver._modelparams.labelAlpha.from_id(labelIdx[i], unknownkey));
	}
	if (output.size() == 0){
		std::cout << "predict error" << std::endl;
	}
 	return 0;
}

void Classifier::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	m_driver.TestInitial();
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts);

	vector<Example> testExamples;
	initialExamples(testInsts, testExamples);

	int testNum = testExamples.size();
	vector<Instance> testInstResults;
	Metric metric_test;
	metric_test.reset();
	for (int idx = 0; idx < testExamples.size(); idx++) {
		vector<string> result_label;
		predict(testExamples[idx], result_label);
		testInsts[idx].evaluate(result_label, metric_test);
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_label);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

}


void Classifier::loadModelFile(const string& inputModelFile) {
	ifstream is(inputModelFile);
	if (is.is_open()) {
		m_driver._hyperparams.loadModel(is);
		m_driver._modelparams.loadModel(is);
		is.close();
	}
	else
		cout << "load model error" << endl;
}

void Classifier::writeModelFile(const string& outputModelFile) {
	ofstream os(outputModelFile);
	if (os.is_open()) {
		m_driver._hyperparams.saveModel(os);
		m_driver._modelparams.saveModel(os);
		os.close();
		cout << "write model ok. " << endl;
	}
	else
		cout << "open output file error" << endl;
}


int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	int memsize = 0;
 	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

	ah.process(argc, argv);

	if (memsize < 0)
		memsize = 0;
	Classifier the_classifier;
	if (bTrain) {
		the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		the_classifier.test(testFile, outputFile, modelFile);
	}
	//getchar();
	//test(argv);
	//ah.write_values(std::cout);
}
//int main(){
//	Classifier the_classifier;
//	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
//	std::string outputFile = "";
//	bool bTrain = false; 
//	int memsize = 0;
//	dsr::Argument_helper ah;
//
//	ah.new_flag("l", "learn", "train or test", bTrain);
//	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
//	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
//	ah.new_named_string("test", "testCorpus", "named_string",
//		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
//	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
//	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
//	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
//	ah.process(argc, argv);
//
//	vector<Instance> trainInsts;
//	std::string s = "D:\\vs_workspace\\FirstSegmentor\\Experiment\\data\\corpus\\test.hwc";
//	//the_classifier.m_pipe.readInstances(s,trainInsts);
//	//the_classifier.createAlphabet(trainInsts);
//	//vector<Example> exam;
//	//
//	//the_classifier.initialExamples(trainInsts,exam);
//	the_classifier.train(s, s, s, s, s);
//
//	int a;
//	cin >> a;
//	return 0;
//}