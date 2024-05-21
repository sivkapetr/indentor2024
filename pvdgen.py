import sys
with open('nonlin_elast_out.pvd', 'w') as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1">\n')
    f.write('  <Collection>\n')
    for i in range(1, int(sys.argv[1])+1):
         f.write('    <DataSet timestep="' + str(i) +'" part="0" file="' + "nonlin_elast_out"  + str(i//10) + str(i%10) + '.pvtu" />\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>')
