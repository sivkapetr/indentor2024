# Программа для генерации сетки четверьти образца.  a, b - размеры по осям x и y. c - высота объекта
# Остальные параметры можно не трогать. n, m - количество отрезков по осям x и y. p - количество отрезков по высоте.
import sys
a = 10e-3#10e-3
b = 10e-3#10e-3
c = 6e-3#5e-3

n = 10
m = 10
p = 2

q = 1.3

xm = 0
ym = 0
x = []
y = []
xi = 0
yi = 0
for i in range(n):
   xm += q**i
for i in range(n+1):
   x.append(xi)
   xi += q**i/xm
for i in range(m):
   ym += q**i
for i in range(m+1):
   y.append(yi)
   yi += q**i/ym
   
with open('mesh.msh', 'w') as f:
    f.write('$MeshFormat\n')
    f.write('2.2 0 8\n')
    f.write('$EndMeshFormat\n')
    f.write('$Nodes\n')
    f.write(str((n+1)*(m+1)*(p+1))+'\n')
    for i in range(n+1):
       for j in range(m+1):
          for k in range(p+1):
             f.write(str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(a*x[i]) + ' ' + str(b*y[j]) + ' ' + str(c*k/(p)) +'\n')
    f.write('$EndNodes\n')
    f.write('$Elements\n')
    f.write(str(n*m*4+2*n*p+2*m*p+6*n*m*p)+'\n')
    for j in range(m):
       for i in range(n):
          if(i%2==0):
             if(j%2==0):
                f.write(str(2*(i+j*m)+1) + ' 2 2 3 3 ' + str(1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                f.write(str(2*(i+j*m)+2) + ' 2 2 3 3 ' + str(1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
             else:
                f.write(str(2*(i+j*m)+1) + ' 2 2 3 3 ' + str(1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+(j)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                f.write(str(2*(i+j*m)+2) + ' 2 2 3 3 ' + str(1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
          else:
             if(j%2==0):
                f.write(str(2*(i+j*m)+1) + ' 2 2 3 3 ' + str(1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                f.write(str(2*(i+j*m)+2) + ' 2 2 3 3 ' + str(1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
             else:
                f.write(str(2*(i+j*m)+1) + ' 2 2 3 3 ' + str(1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                f.write(str(2*(i+j*m)+2) + ' 2 2 3 3 ' + str(1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
    for j in range(m):
       for i in range(n):
          if(j%2==0):
             if(i%2==0):
                f.write(str(n*m*2+2*(i+j*m)+1) + ' 2 2 4 4 ' + str(1+p+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                f.write(str(n*m*2+2*(i+j*m)+2) + ' 2 2 4 4 ' + str(1+p+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+p+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
             else:
                f.write(str(n*m*2+2*(i+j*m)+1) + ' 2 2 4 4 ' + str(1+p+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                f.write(str(n*m*2+2*(i+j*m)+2) + ' 2 2 4 4 ' + str(1+p+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
          else:
             if(i%2==0):
                f.write(str(n*m*2+2*(i+j*m)+1) + ' 2 2 4 4 ' + str(1+p+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+p+(j)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+p+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                f.write(str(n*m*2+2*(i+j*m)+2) + ' 2 2 4 4 ' + str(1+p+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
             else:
                f.write(str(n*m*2+2*(i+j*m)+1) + ' 2 2 4 4 ' + str(1+p+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                f.write(str(n*m*2+2*(i+j*m)+2) + ' 2 2 4 4 ' + str(1+p+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+p+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+p+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
    for j in range(p):
       for i in range(n):
         if(i%2==0):
          if(j%2==0):
             f.write(str(n*m*4+2*(i+j*n)+1) + ' 2 2 2 2 ' + str(1+j+i*(p+1)*(m+1)) + ' ' + str(1+j+(i+1)*(p+1)*(m+1)) + ' ' + str(2+j+(i+1)*(p+1)*(m+1))+'\n')
             f.write(str(n*m*4+2*(i+j*n)+2) + ' 2 2 2 2 ' + str(1+j+i*(p+1)*(m+1)) + ' ' + str(2+j+i*(p+1)*(m+1)) + ' ' + str(2+j+(i+1)*(p+1)*(m+1))+'\n')
          else:
             f.write(str(n*m*4+2*(i+j*n)+1) + ' 2 2 2 2 ' + str(1+j+1+i*(p+1)*(m+1)) + ' ' + str(1+j+1+(i+1)*(p+1)*(m+1)) + ' ' + str(1+j+(i+1)*(p+1)*(m+1))+'\n')
             f.write(str(n*m*4+2*(i+j*n)+2) + ' 2 2 2 2 ' + str(1+j+1+i*(p+1)*(m+1)) + ' ' + str(1+j+i*(p+1)*(m+1)) + ' ' + str(1+j+(i+1)*(p+1)*(m+1))+'\n')
         else:
          if(j%2==0):
             f.write(str(n*m*4+2*(i+j*n)+1) + ' 2 2 2 2 ' + str(1+j+(i+1)*(p+1)*(m+1)) + ' ' + str(1+j+(i)*(p+1)*(m+1)) + ' ' + str(2+j+(i)*(p+1)*(m+1))+'\n')
             f.write(str(n*m*4+2*(i+j*n)+2) + ' 2 2 2 2 ' + str(1+j+(i+1)*(p+1)*(m+1)) + ' ' + str(2+j+(i+1)*(p+1)*(m+1)) + ' ' + str(2+j+(i)*(p+1)*(m+1))+'\n')
          else:
             f.write(str(n*m*4+2*(i+j*n)+1) + ' 2 2 2 2 ' + str(1+j+1+(i+1)*(p+1)*(m+1)) + ' ' + str(1+j+1+(i)*(p+1)*(m+1)) + ' ' + str(1+j+(i)*(p+1)*(m+1))+'\n')
             f.write(str(n*m*4+2*(i+j*n)+2) + ' 2 2 2 2 ' + str(1+j+1+(i+1)*(p+1)*(m+1)) + ' ' + str(1+j+(i+1)*(p+1)*(m+1)) + ' ' + str(1+j+(i)*(p+1)*(m+1))+'\n')
    for j in range(p):
       for i in range(m):
         if(i%2==0):
          if(j%2==0):
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+1) + ' 2 2 1 1 ' + str(1+j+i*(p+1)) + ' ' + str(1+j+(i+1)*(p+1)) + ' ' + str(2+j+(i+1)*(p+1))+'\n')
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+2) + ' 2 2 1 1 ' + str(1+j+i*(p+1)) + ' ' + str(2+j+i*(p+1)) + ' ' + str(2+j+(i+1)*(p+1))+'\n')
          else:
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+1) + ' 2 2 1 1 ' + str(2+j+i*(p+1)) + ' ' + str(2+j+(i+1)*(p+1)) + ' ' + str(1+j+(i+1)*(p+1))+'\n')
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+2) + ' 2 2 1 1 ' + str(2+j+i*(p+1)) + ' ' + str(1+j+i*(p+1)) + ' ' + str(1+j+(i+1)*(p+1))+'\n')
         else:
          if(j%2==0):
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+1) + ' 2 2 1 1 ' + str(1+j+(i+1)*(p+1)) + ' ' + str(1+j+(i)*(p+1)) + ' ' + str(2+j+(i)*(p+1))+'\n')
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+2) + ' 2 2 1 1 ' + str(1+j+(i+1)*(p+1)) + ' ' + str(2+j+(i+1)*(p+1)) + ' ' + str(2+j+(i)*(p+1))+'\n')
          else:
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+1) + ' 2 2 1 1 ' + str(2+j+(i+1)*(p+1)) + ' ' + str(2+j+(i)*(p+1)) + ' ' + str(1+j+(i)*(p+1))+'\n')
             f.write(str(n*m*4+2*n*p+2*(i+j*m)+2) + ' 2 2 1 1 ' + str(2+j+(i+1)*(p+1)) + ' ' + str(1+j+(i+1)*(p+1)) + ' ' + str(1+j+(i)*(p+1))+'\n')
    for i in range(n):
       for j in range(m):
          for k in range(p):
            if(i%2==0):
              if(j%2==0):   
                if(k%2==0): 
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                else:
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
              else:
                if(k%2==0): 
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                else:
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+i*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   
                   
            else:
              if(j%2==0):   
                if(k%2==0): 
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                else:
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+j*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+j*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
              else:
                if(k%2==0): 
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                else:
                   f.write(str(n*m*4+2*n*p+2*m*p+1+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+2+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+3+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+4+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+1+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+5+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
                   f.write(str(n*m*4+2*n*p+2*m*p+6+(k+j*p+i*p*m)*6) + ' 4 2 1 1 ' + str(1+k+1+(j+1)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i+1)*(p+1)*(m+1)) + ' ' + str(1+k+(j)*(p+1)+(i)*(p+1)*(m+1)) + ' ' + str(1+k+(j+1)*(p+1)+(i+1)*(p+1)*(m+1))+'\n')
    f.write('$EndElements\n')
