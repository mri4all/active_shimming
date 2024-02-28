import cadquery as cq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def export_magnet_ledger_to_cad(xlsfilename,templatefile):
    assy = cq.Assembly()
    
    ledger = pd.read_excel(xlsfilename, 
               dtype={'X-pos': float, 
                      'Y-pos': float, 
                      'Z-pos': float, 
                      'X-rot': float, 
                      'Y-rot': float,
                      'Z-rot': float,
                      'Coil-diam': float,
                      'r': float,
                      'number_mags_aximuthal': float,
                      'coil_diam': float,
                      'delta_z': float,
                      'delta_theta': float,
                      'nz': float,
                      'ref_curr': str})   
    
    ledger.columns = ['X-pos','Y-pos','Z-pos','X-rot','Y-rot','Z-rot','Searched','CostValue','Used','Placement_index','Bmag','Magnet_length','Tag']
    ledger = ledger.reset_index(drop=True) #needed correction since rows are dropped

    for idx_ledger, ledger_row in ledger.iterrows():
        print('iterating index: ' + str(idx_ledger) + 'out of: ' + str(ledger.shape[0]))
        # cube_w_margine = (ledger_row['Magnet_length'],
        #                   ledger_row['Magnet_length'],
        #                   ledger_row['Magnet_length'])
        # if(idx_ledger==0):
        # result = cq.Workplane("YZ").box(cube_w_margine[0], cube_w_margine[1], cube_w_margine[2]).circle(0.5).extrude(cube_w_margine[0]/2+1).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))

# rotating in the Z, Y, and X directions in this order. doing this at the center and then translating to a position


# LA commented out 9.20.23---------------------------
        # result = cq.Workplane("YZ").box(cube_w_margine[0], cube_w_margine[1], cube_w_margine[2]).circle(0.5).extrude(cube_w_margine[0]/2+1).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))
        
        result = cq.importers.importStep(templatefile).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))
        # else:
        #     result2 = cq.Workplane("YZ").box(cube_w_margine[0], cube_w_margine[1], cube_w_margine[2]).circle(0.5).extrude(cube_w_margine[0]/2+1).rotateAboutCenter((1, 0, 0), ledger_row['X-rot']).rotateAboutCenter((0, 1, 0), ledger_row['Y-rot']).rotateAboutCenter((0, 0, 1), ledger_row['Z-rot']).translate((ledger_row['X-pos'],ledger_row['Y-pos'],ledger_row['Z-pos']))
        #     result = result.union(result2)
            
        
    # cq.exporters.export(result, filename[:-4]+ 'step')
        assy.add(result,color=cq.Color("red"))

    assy.save(xlsfilename[:-4]+ 'step')
     
# smaller_thickness = 7.0
# larger_thickness = 3
# center_hole_dia = 8.0

# # Create a box based on the dimensions above and add a 22mm center hole
# result = (
#     cq.Workplane("YZ")
#     .circle(10)
#     .extrude(smaller_thickness)
#     .faces(">X")
#     .workplane()
#     .circle(20)
#     .extrude(larger_thickness)
# ).rotateAboutCenter((1, 0, 0), 0).rotateAboutCenter((0, 1, 0), 0).rotateAboutCenter((0, 0, 1), 0).translate((-(larger_thickness
# +smaller_thickness)/2,0,0))

# show_object(result)


