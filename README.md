A Full Implementation of an Encoder-Decoder AI Transformer in C/C++

https://www.youtube.com/watch?v=iDcndq3XEIs

https://forejune.co/cuda/

A Hello-World Example of AI Transformer in C/C++ and CUDA   Youtube

https://forejune.co/cuda/ai/transform/transform.html

Explaing an AI Transformer Components in C/C++ (Part 1)   Youtube

https://forejune.co/cuda/ai/transform2/transform2.html

Explaing an AI Transformer Components in C/C++ (Part 2)   Youtube

https://forejune.co/cuda/ai/transform2/transform2a.html

Decoder of an AI Transformer in C/C++ Youtube

https://forejune.co/cuda/ai/transform3/transform3.html

Backpropagation of an AI Transformer in C/C++   Youtube

https://forejune.co/cuda/ai/transform4/transform4.html

Full Implementation of an AI Transformer in C/C++   Youtube

https://forejune.co/cuda/ai/transform5/transform5.html

Full Encoder-Decoder Transformer in C/C++   Youtube

https://forejune.co/cuda/ai/transform6/transform6.html


ly@xuan:~/pulp_box/forejune_co_cuda/transformer6$ make clean

rm testMain.o util.o  transformer.o testMain

ly@xuan:~/pulp_box/forejune_co_cuda/transformer6$ make testMain

g++ -c  -std=c++20 testMain.cpp

g++ -c  -std=c++20 util.cpp

g++ -c  -std=c++20 transformer.cpp

g++ -o testMain testMain.o util.o  transformer.o

ly@xuan:~/pulp_box/forejune_co_cuda/transformer6$ ./testMain

Vocab size: 33

==== Encoder-Decoder Transformer Tests ====

Example 1:

Epoch 0 Loss: 3.41549

Epoch 50 Loss: 3.12469

Epoch 100 Loss: 2.85001

Epoch 150 Loss: 2.58962

Epoch 200 Loss: 2.34093

Epoch 250 Loss: 2.10141

Epoch 300 Loss: 1.86959

Epoch 350 Loss: 1.64597

Epoch 400 Loss: 1.43323

Epoch 450 Loss: 1.23523

Epoch 500 Loss: 1.05567

Epoch 550 Loss: 0.897

Epoch 600 Loss: 0.759997


==== Inference ====

Source: one two three four

Predicted: uno dos tres cuatro

Expected:  uno dos tres cuatro



Example 2:

Epoch 0 Loss: 3.8202

Epoch 50 Loss: 3.17425

Epoch 100 Loss: 2.58446

Epoch 150 Loss: 2.14866

Epoch 200 Loss: 1.80427

Epoch 250 Loss: 1.49905

Epoch 300 Loss: 1.22449

Epoch 350 Loss: 0.984851

Epoch 400 Loss: 0.783814

Epoch 450 Loss: 0.621525

Epoch 500 Loss: 0.494763

Epoch 550 Loss: 0.397971

Epoch 600 Loss: 0.324794


==== Inference ====

Source: one two three four reverse

Predicted: four three two one

Expected:  four three two one


Example 3:

Epoch 0 Loss: 3.84994

Epoch 50 Loss: 2.8222

Epoch 100 Loss: 2.1992

Epoch 150 Loss: 1.75621

Epoch 200 Loss: 1.39235

Epoch 250 Loss: 1.09029

Epoch 300 Loss: 0.848987

Epoch 350 Loss: 0.662503

Epoch 400 Loss: 0.520799

Epoch 450 Loss: 0.414033

Epoch 500 Loss: 0.333977

Epoch 550 Loss: 0.27392

Epoch 600 Loss: 0.228525


==== Inference ====

Source: one two three four digit

Predicted: 1 2 3 4

Expected:  1 2 3 4

