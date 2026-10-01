//~/pulp_box/forejune_co_cuda/transformer6/testMain.cpp //https://forejune.co/cuda/
#include <cmath>
#include <algorithm>
#include <fstream>
#include "util.h"
#include "transformer.h"
using namespace std;
struct TrainingDS {
    vector<int> source;
    vector<int> target_in;
    vector<int> target_out;
};
vector <TrainingDS> buildDataset(Tokenizer &tokenizer){
   vector <TrainingDS> dataset;
   TrainingDS ds;  
   const int n = 3;  // 3 training examples
   string sources[n] = { "one two three four", "one two three four reverse", "one two three four digit"};
   string target_ins[n] = { "<BOS> uno dos tres cuatro", "<BOS> four three two one", "<BOS> 1 2 3 4"};
   string target_outs[n] = { "uno dos tres cuatro <EOS>", "four three two one <EOS>", "1 2 3 4 <EOS>"};
   for (int i = 0; i < n; i++) {
      vector<int> token_src = tokenizer.tokenize(sources[i]);
      vector<int> token_tin = tokenizer.tokenize(target_ins[i] );
      vector<int> token_tout =tokenizer.tokenize(target_outs[i]); 
      ds = TrainingDS(token_src, token_tin, token_tout);
      dataset.push_back( ds );
   }
   return dataset;
}
int main(){
    // Tokenizer setup  // Build vocabulary 
    ifstream ifs;
    char fname[] = "t.txt";
    ifs.open( fname );
    if (!ifs.is_open()) { cerr << "Unable to open file " << fname  << endl; exit( 1 ); }
    Tokenizer tokenizer( ifs );
    int vocab_size = tokenizer.get_nTokens();
    cout << "Vocab size: " << vocab_size << endl;
    vector <TrainingDS> datasets;
    datasets = buildDataset( tokenizer );
    int n_ds = datasets.size(); 
    cout << "==== Encoder-Decoder Transformer Tests ====" << endl;
    // Hyperparameters
    int seq_len = 16;      
    int dModel = 16;
    int nHeads = 4;
    int d_ff = 16; 
    int nLayers = 4;
    double eta = 0.002;
    int nEpoch = 601;
    // Model
    MiniTrans model(vocab_size, seq_len, dModel, nHeads, d_ff, nLayers);
    int START = tokenizer.tokenize( "<BOS>" )[0];
    int EOS = tokenizer.tokenize( "<EOS>" )[0];
    for (int k = 0; k < n_ds; k++) {
       cout << "Example " << k + 1 << ":" << endl;
       // Training loop
       for (int i = 0; i < nEpoch; i++) { double loss;
           loss = model.trainStep(datasets[k].source, datasets[k].target_in, 
           datasets[k].target_out, eta);
           if (i % 50 == 0)  cout << "Epoch " << i << " Loss: " << loss << endl;
       }
       // Inference (greedy decoding)
       cout << "\n==== Inference ====" << endl;
       vector<int> predict = model.generate(datasets[k].source, 12, START, EOS);
       cout << "Source: ";
       string str = tokenizer.detokenize( datasets[k].source );
       cout << str << endl;
       cout << "Predicted: ";
       str = tokenizer.detokenize( predict );
       cout << str << endl; 
       cout << "Expected:  ";
       str = tokenizer.detokenize( datasets[k].target_out );
       cout << str << endl; 
       getchar();   // pause
    }  // for k
    return 0;
}
