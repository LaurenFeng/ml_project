{\rtf1\ansi\ansicpg1252\cocoartf2636
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0  kwargs = \{'Training X': tr_X,\
              'Training Y': tr_Y,\
              'Test X': te_X,\
              'Test Y': te_Y,\
              'max_iters': 900,\
              'Learning rate': 0.001,\
              'Weight decay': 0.0001,\
              'Mini-batch size': 300,\
              'record_every': 10,\
              'Test loss function name': '0-1 error',\
              'Feature map filename': '../data/feature_maps.pkl'\
              \}\
 dimensions = [input_dim, 2048,1024,512, num_classes]\
 activation_funcs = \{1:ReLU, 2:ReLU, 3:ReLU,4:Softmax\}\
\
\
M1 cord, 2:30, accuracy=94%}