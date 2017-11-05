#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
public:
	const static int max_sentence_length = 1024;

public:
	// node instances
	

	
	vector<LookupNode> _char_inputs; //
	LSTM1Builder _lstm;
	vector<UniNode> _character;
	vector<LinearNode> _neural_outputs;

	Graph *_pcg;

public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_char_inputs.resize(sent_length);
		_lstm.resize(sent_length); 
		_character.resize(sent_length);
		_neural_outputs.resize(sent_length);
	}

	inline void clear(){
		_char_inputs.clear();
		_lstm.clear();
		_neural_outputs.clear();
	}

public:
	inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts){
		_pcg = pcg;
		for (int idx = 0; idx < _char_inputs.size(); idx++) {
			_char_inputs[idx].setParam(&model.chars);
			_char_inputs[idx].init(opts.charDim,opts.dropProb);
		}

		_lstm.init(&model.lstm_param, opts.dropProb, true);
		
		for (int i = 0; i < _character.size(); i++) {
			_character[i].setParam(&model.character);
			_character[i].init(opts.characterSize, opts.dropProb);

		}
		for (int i = 0; i < _neural_outputs.size(); i++) {
		
			_neural_outputs[i].setParam(&model.olayer_linear);
			_neural_outputs[i].init(opts.labelSize, -1);
		}


		//_neural_output.setParam(&model.olayer_linear);
		//_neural_output.init(opts.labelSize, -1);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Example exam, bool bTrain = false){
		_pcg->train = bTrain;
		// second step: build graph
		//forward 

		int char_num = exam.m_chars.size();
		if (char_num > max_sentence_length)
			char_num = max_sentence_clength;
		for (int i = 0; i < char_num; i++) {
			_char_inputs[i].forward(_pcg,exam.m_chars[i]);
		}
		_lstm.forward(_pcg, getPNodes(_char_inputs,char_num));

		for (int i = 0; i < char_num; i++) {
			_character[i].forward(_pcg, &_lstm._hiddens[i]);
		}

		
		for (int i = 0; i < char_num; i++) {
			_neural_outputs[i].forward(_pcg, &_character[i]);
		}

	}
		//_avg_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		//_max_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		//_min_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		//_concat.forward(_pcg, &_avg_pooling, &_max_pooling, &_min_pooling);
		
};

#endif /* SRC_ComputionGraph_H_ */