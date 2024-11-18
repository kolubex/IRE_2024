# Code Generation Web App

#### Directory Structure
```
.
├── app.ipynb
├── app.py
├── code_generator.py
├── evaluate_oops.py
├── gen_descr.py
├── generate_mbpp_results.py
├── gen_gold.py
├── gen_results_oops.py
├── make_vector_db.py
├── mbpp_eval.py
├── mbpp_gen_codes.py
└── README.md

1 directory, 12 files

```

#### To launch public app:
```
python app.py
```

#### To generating results for OOPs benchmark:
```
python3 evaluate_oops.py
```

#### To generate descriptions for OOPs benchmark:
```
python3 gen_descr.py
```
#### To generate code for MBPP benchmark:
```
python3 generate_mbpp_results.py
```

#### To generate Gold Standard Code for OOPs benchmark using Mixtral 8x7b:
```
python3 gen_gold.py
```

#### To Make Vector DB:
```
python3 make_vector_db.py
```
#### To evaluate generated code for MBPP benchmark:
```
python3 mbpp_eval.py
```
#### To view the Code Generator Class, open:
```
code_generator.py
```
