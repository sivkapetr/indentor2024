# indentor2024
indentation test for 2024 school

В этом репозитории хранится код на языке cpp с применением библиотеки inmost-fem.
Поэтому рекомендуется начать именно с установки [библиотеки:](https://github.com/INMOST-DEV/INMOST-FEM)

С этой целью можно воспользоваться руководством из репозитория inmost-fem или выполнить последовательно следующие шаги:

```console
git clone https://github.com/INMOST-DEV/INMOST-FEM.git

cd INMOST-FEM

mkdir build

cd build

cmake -DWITH_INMOST=ON -DDOWNLOAD_inmost=ON -DWITH_KINSOL=ON 
-DDOWNLOAD_sundials=ON -DWITH_EIGEN=ON -DDOWNLOAD_eigen3=ON 
-DWITH_CASADI=ON -DDOWNLOAD_casadi=ON -DCOMPILE_EXAMPLES=ON 
-DCOMPILE_TESTS=ON ../

cmake --build .

cmake -DCMAKE_INSTALL_PREFIX=/home/student/libs/anifem++_install ..

cmake --install .
```
