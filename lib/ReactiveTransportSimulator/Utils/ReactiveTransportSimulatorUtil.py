import json
import os
import subprocess
import h5py
import uuid
from installed_clients.KBaseReportClient import KBaseReport
from installed_clients.DataFileUtilClient import DataFileUtil
from pprint import pprint
from shutil import copy
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import json

class ReactiveTransportSimulatorUtil:
    PREPDE_TOOLKIT_PATH = '/kb/module/lib/ReactiveTransportSimulator/Utils'

    def _generate_html_report(self):
        report = "<html> <head> ReactiveTransportSimulator-KBase report </head> <body> </body> </html>"
        return report 

class ReactiveTransportSimulatorRunBatchUtil:
    def __init__(self,params):
        self.params = params
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.dfu = DataFileUtil(self.callback_url)
        self.output_files = []
        self.html_files = []
        self.data_folder = os.path.abspath('./data/')
        self.shared_folder = params['shared_folder']
        self.scratch_folder = os.path.join(params['shared_folder'],"scratch")

    def run_batch_model(self):

        print('params:',self.params)
        try:
            os.mkdir(self.scratch_folder)
        except OSError:
            print ("Creation of the directory %s failed" % self.scratch_folder)
        else:
            print ("Successfully created the directory %s " % self.scratch_folder)

        # move file templates from data folder to scratch folder 
        pflotran_input_temp = os.path.join(self.data_folder,'batch_template.in')
        pflotran_db_temp    = os.path.join(self.data_folder,'database_template.dat')
        pflotran_input      = os.path.join(self.scratch_folder,'batch.in')
        pflotran_db         = os.path.join(self.scratch_folder,'database.dat')
        stoi_csv_fba        = os.path.join(self.scratch_folder,'rxn_fba.csv')
        cpd_csv_fba         = os.path.join(self.scratch_folder,'cpd_fba.csv')

        # read inputs
        print("Input FBA model: ",self.params['input_FBA_model'])
        dfu                 = DataFileUtil(self.callback_url)
        fba_model           = dfu.get_objects({'object_refs': [self.params['input_FBA_model']]})['data'][0]
        print("FBA model name :",fba_model['data']['name'])
        nrxn                = int(self.params['number_simulated_reactions'])
        tot_time            = float(self.params['simulation_time'])
        timestep            = float(self.params['snapshot_period'])
        temperature         = float(self.params['temperature'])

        # collect the compound info
        cpdid2formula = dict()
        df_cpd = pd.DataFrame({'formula':[None]})
        for compound in fba_model['data']['modelcompounds']:
            cpdid2formula[compound['id']] = compound['formula']
            if 'biom' in compound['id']:
                df_cpd = df_cpd.append({'formula':'BIOMASS'}, ignore_index=True)
            else:
                df_cpd = df_cpd.append({'formula':compound['formula']}, ignore_index=True)
        df_cpd.insert(len(df_cpd.columns),'initial_concentration(mol/L)',1,True)
        df_cpd['formula'].replace('', np.nan, inplace=True)
        df_cpd = df_cpd.dropna()
        df_cpd.to_csv(cpd_csv_fba,index=False)
        print("Compounds saved. \n")
        
        # collect donor, acceptor, biom from reactions
        """
            donor : "~/modelcompounds/id/xcpd2_c0"
            acceptor : "~/modelcompounds/id/acceptor_c0"
            biom : "~/modelcompounds/id/biom_c0"
        """
        
        rxn_ref = ['r'+str(i+1) for i in range(nrxn)]
        df_rxn = pd.DataFrame({'rxn_ref':rxn_ref,'rxn_id':None,'DOC_formula':None})

        # selected_reactions = random.choices(fba_model['data']['modelreactions'],k=nrxn)
        selected_reactions = []
        selected_cpd       = []
        i = 0
        while i < nrxn:
            irxn = random.choice(fba_model['data']['modelreactions'])
            acceptor_flag = False
            for reagent in irxn['modelReactionReagents']:
                cpdid = reagent['modelcompound_ref'].split('/id/')[1]
                if 'acceptor' in cpdid:
                    acceptor_flag = True
                if 'xcpd' in cpdid:
                    doc = cpdid2formula[cpdid]
                    selected_cpd.append(doc)      
            if acceptor_flag and selected_cpd.count(doc) == 1:    
                selected_reactions.append(irxn)
                i += 1

        for reaction_idx,reaction_val in enumerate(selected_reactions):
            df_rxn['rxn_id'].iloc[reaction_idx] = reaction_val['id']
            for reagent in reaction_val['modelReactionReagents']:
                cpdid = reagent['modelcompound_ref'].split('/id/')[1]
                formula = cpdid2formula[cpdid]
                coef    = reagent['coefficient']

                if "xcpd" in cpdid:
                    df_rxn['DOC_formula'].iloc[reaction_idx] = formula

                if "biom" in cpdid:
                    formula = 'BIOMASS'

                if not formula in df_rxn.columns:
                    temp = ['0']*df_rxn.shape[0]
                    df_rxn.insert(len(df_rxn.columns),formula,temp,True)
                    df_rxn[formula].iloc[reaction_idx] = coef
                else:
                    df_rxn[formula].iloc[reaction_idx] = coef

        print(df_rxn.columns)
        print(df_rxn.head())
        df_rxn.to_csv(stoi_csv_fba,index=False)
        print("Selected reactions saved. \n")


        # read initial condition from /bin/module/data
        init_cond = cpd_csv_fba

        # generate sandbox file
        sb_file = os.path.join(self.scratch_folder,'reaction_sandbox_pnnl_cyber.F90')
        var = ['mu_max','vh','k_deg','cc','activation_energy','reference_temperature']
        var_unit = ['1/sec','m^3','1/sec','M','J/mol','K']
        generate_sandbox_code(nrxn,var,var_unit,sb_file,stoi_csv_fba)
        print("Sandbox file generated.")

        # format sandbox fortran code
        fmt_sb_cmd = 'fprettify ' + sb_file
        process = subprocess.Popen(fmt_sb_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Sandbox file formatted.")

        # copy sandbox file to src dir and recompile pflotran
        src_dir = '/bin/pflotran/src/pflotran'
        copy(sb_file,src_dir)
        print(os.getcwd())
        compile_pflotran_cmd = 'sh ./data/compile.sh'
        process = subprocess.Popen(compile_pflotran_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Compile PFLOTRAN output:",output[-300:])
        print("Complile PFLOTRAN err:",error)
        pprint(os.listdir(self.scratch_folder))

        # generate batch input deck
        self.generate_pflotran_input_batch(pflotran_input_temp,stoi_csv_fba,cpd_csv_fba,pflotran_input,tot_time,timestep,temperature)
        print("Batch input deck generated.")

        # generate database 
        update_pflotran_database(stoi_csv_fba,pflotran_db_temp,pflotran_db)
        print("Database generated.")

        # running pflotran
        exepath = '/bin/pflotran/src/pflotran/pflotran'
        run_pflotran_cmd = exepath + ' -n 1 -pflotranin ' + pflotran_input
        process = subprocess.Popen(run_pflotran_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Running PFLOTRAN output:",output[-300:])
        print("Running PFLOTRAN err:",error)
        pprint(os.listdir(self.scratch_folder))

        h5_file = os.path.join(self.scratch_folder,'batch.h5')
        if os.path.isfile(h5_file):
            print ("Successfully run PFLOTRAN")
        else:
            print ("Fail to run PFLOTRAN")

        # generate plots in /kb/module/work/tmp/scratch/
        self.plot_time_series_batch(h5_file)
        
        
        # Attach output
        self.output_files.append(
            {'path': cpd_csv_fba,
             'name': os.path.basename(cpd_csv_fba),
             'label': os.path.basename(cpd_csv_fba),
             'description': 'compounds'}
        ) 
        self.output_files.append(
            {'path': stoi_csv_fba,
             'name': os.path.basename(stoi_csv_fba),
             'label': os.path.basename(stoi_csv_fba),
             'description': 'reactions stoichiometry table'}
        )      
        self.output_files.append(
            {'path': sb_file,
             'name': os.path.basename(sb_file),
             'label': os.path.basename(sb_file),
             'description': 'Sandbox source code'}
        )              
        self.output_files.append(
            {'path': pflotran_input,
             'name': os.path.basename(pflotran_input),
             'label': os.path.basename(pflotran_input),
             'description': 'Batch reaction input deck for PFLOTRAN'}
        )  
        self.output_files.append(
            {'path': pflotran_db,
             'name': os.path.basename(pflotran_db),
             'label': os.path.basename(pflotran_db),
             'description': 'Batch reaction input deck for PFLOTRAN'}
        )  
        self.output_files.append(
            {'path': h5_file,
             'name': os.path.basename(h5_file),
             'label': os.path.basename(h5_file),
             'description': 'H5 file generated by PFLOTRAN batch reaction'}
        )
        fig_name = 'time_series_plot.png'
        fig_file = os.path.join(self.scratch_folder,fig_name) 
        self.output_files.append(
            {'path': fig_file,
             'name': os.path.basename(fig_file),
             'label': os.path.basename(fig_file),
             'description': 'Plots of breakthrough curves generated by PFLOTRAN batch reaction'}
        )

        # Return the report
        return self._generate_html_report()

    def generate_pflotran_input_batch(self,batch_file,stoi_file,init_file,output_file,tot_time,timestep,temp):
        file = open(batch_file,'r')
        rxn_df = pd.read_csv(stoi_file)
        init_df = pd.read_csv(init_file)

        primary_species_charge = []
        primary_species_nocharge = []
        for spec in list(rxn_df.columns):
            if spec in ['rxn_id','DOC_formula','rxn_ref','H2O','BIOMASS']:
                continue
            primary_species_nocharge.append(spec)
            if spec=='NH4':
                primary_species_charge.append('NH4+')
                continue
            if spec=='HCO3':
                primary_species_charge.append('HCO3-')
                continue
            if spec=='H':
                primary_species_charge.append('H+')
                continue
            if spec=='HS':
                primary_species_charge.append('HS-')
                continue
            if spec=='HPO4':
                primary_species_charge.append('HPO4-')  
                continue
            primary_species_charge.append(spec) 

        init_cond = [init_df.loc[init_df['formula']==i,'initial_concentration(mol/L)'].iloc[0] for i in primary_species_nocharge]
        init_biom = init_df.loc[init_df['formula']=='BIOMASS','initial_concentration(mol/L)'].iloc[0]
        for idx,val in enumerate(primary_species_nocharge):
            print("The initial concentration of {} is {} mol/L \n".format(val,init_cond[idx]))

        pri_spec = ""
        pri_spec_init = ""
        new_file_content = ""
        for line in file:           
            if 'PRIMARY_SPECIES' in line:
                new_file_content += line
                for i in primary_species_charge:
                    pri_spec += "    " + i + "\n"  
                new_file_content += "    " + pri_spec + "\n" 

            elif 'CONSTRAINT initial' in line:
                new_file_content += line
                new_file_content += "  CONCENTRATIONS" + "\n"
                for j in range(len(primary_species_charge)):
                    new_file_content += "    {}        {} T".format(primary_species_charge[j],init_cond[j])+ "\n"
                new_file_content += "  /" + "\n"
                new_file_content += "  IMMOBILE" + "\n"
                new_file_content += "    BIOMASS        {} ".format(init_biom) + "\n"
                new_file_content += "  /"   

            elif 'FINAL_TIME' in line:
                new_file_content += "  FINAL_TIME {} h".format(tot_time) + "\n"
                
            elif 'FINAL_TIME' in line:
                new_file_content += "  FINAL_TIME {} h".format(tot_time) + "\n"
                
            elif 'MAXIMUM_TIMESTEP_SIZE' in line:
                new_file_content += "  MAXIMUM_TIMESTEP_SIZE {} h".format(timestep) + "\n"
                
            elif 'PERIODIC TIME' in line:
                new_file_content += "    PERIODIC TIME {} h".format(timestep) + "\n"        
                
            elif 'REFERENCE_TEMPERATURE' in line:
                new_file_content += "      REFERENCE_TEMPERATURE {} ! degrees C".format(temp) + "\n"
                
            else:
                new_file_content += line  
                
        writing_file = open(output_file, "w")
        writing_file.write(new_file_content)
        writing_file.close()
        print('The batch input deck is updated.')
        return

    def plot_time_series_batch(self,h5_file):
        obs_coord = [0.5,0.5,0.5]

        file = h5py.File(h5_file,'r+')
        time_str = [list(file.keys())[i] for i in range(len(list(file.keys()))) if list(file.keys())[i][0:4] == "Time"]
        time_unit = time_str[0][-1]
        time = sorted([float(time_str[i].split()[1]) for i in range(len(time_str))])
        bound = []
        bound.append(file['Coordinates']['X [m]'][0])
        bound.append(file['Coordinates']['X [m]'][-1])
        bound.append(file['Coordinates']['Y [m]'][0])
        bound.append(file['Coordinates']['Y [m]'][-1])
        bound.append(file['Coordinates']['Z [m]'][0])
        bound.append(file['Coordinates']['Z [m]'][-1])
        nxyz = []
        nxyz.append(len(file['Coordinates']['X [m]'])-1)
        nxyz.append(len(file['Coordinates']['Y [m]'])-1)
        nxyz.append(len(file['Coordinates']['Z [m]'])-1)

        x_coord = (np.linspace(bound[0],bound[1],nxyz[0]+1)[:-1]+np.linspace(bound[0],bound[1],nxyz[0]+1)[1:])/2
        y_coord = (np.linspace(bound[2],bound[3],nxyz[1]+1)[:-1]+np.linspace(bound[2],bound[3],nxyz[1]+1)[1:])/2
        z_coord = (np.linspace(bound[4],bound[5],nxyz[2]+1)[:-1]+np.linspace(bound[4],bound[5],nxyz[2]+1)[1:])/2
        x_idx = np.argmin(np.absolute(x_coord-obs_coord[0]))
        y_idx = np.argmin(np.absolute(y_coord-obs_coord[1]))
        z_idx = np.argmin(np.absolute(z_coord-obs_coord[2]))
        time_zero = "Time:"+str(" %12.5E" % 0)+str(" %s" % time_unit)
        var_name = [x for x in list(file[time_zero].keys()) if 'Total' in x]
        var_value = np.zeros((len(var_name),len(time)))
        for i, itime in enumerate(time):
            time_slice = "Time:"+str(" %12.5E" % itime)+str(" %s" % time_unit)
        #     print(file[time_slice][var_name].keys())
            for j in range(len(var_name)):
                var_value[j,i] = file[time_slice][var_name[j]][x_idx][y_idx][z_idx]

        fig = plt.figure(num=1,dpi=150)
        first_doc = True
        for i in range(len(var_name)):
            if var_name[i][6] == 'C':
                if first_doc == True:
                    plt.plot(time,var_value[i,:],label='DOCs',color='k')[0]
                    first_doc = False
                else:
                    plt.plot(time,var_value[i,:],color='k')[0]
            else:
                plt.plot(time,var_value[i,:],label=var_name[i])[0]
            plt.ioff()

        plt.xlabel("Time (%s)" %time_unit)
        ylabel = 'Concentration [M]'
        plt.ylabel(ylabel)
        plt.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)
        fig_name = 'time_series_plot.png'
        fig_path = os.path.join(self.scratch_folder,fig_name)    
        plt.savefig(fig_path,dpi=150,bbox_inches='tight') 

        if os.path.isfile(fig_path):
            print ("Successfully generated time series plot")
        else:
            print ("Fail to generate time series plot")

        return

    def visualize_hdf_in_html(self):
        output_directory = os.path.join(self.shared_folder,'output') 
        os.makedirs(output_directory)
        print("output dir:", output_directory)
        html_file = os.path.join(output_directory,'summary.html')
        fig_name = 'time_series_plot.png'
        pflotran_out_name = 'batch.out'
        fig_path = os.path.join(self.scratch_folder,fig_name)
        pflotran_out_path = os.path.join(self.scratch_folder,pflotran_out_name)
        if os.path.isfile(fig_path):
            print ("Time series plot exists")
        else:
            print ("Time series plot does not exist")
        print("figpath:",fig_path)
        if os.path.isfile(pflotran_out_path):
            print ("PFLOTRAN output exists")
        else:
            print ("PFLOTRAN output does not exist")
        print("figpath:",pflotran_out_path)

        copy(fig_path,'/kb/module/work/tmp/output')
        copy(pflotran_out_path,'/kb/module/work/tmp/output')
        with open(html_file, 'w') as f:
            f.write("""
                <!DOCTYPE html>
                <html>
                <body>

                <h1>PFLOTRAN-KBbase</h1>
                <p>PFLOTRAN output</p>
                <embed src="batch.out" width="480" height="960">
                <p>Visulize PFLOTRAN output</p>
                <img src="{}" alt="Time series plot" height="360" width="480"></img>
                </body>
                </html>
            """.format(fig_name))


        with open(html_file, 'r') as f:
            print("html_file:",f.readlines())

        report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,
                                                  'pack': 'zip'})['shock_id']

        return {'shock_id': report_shock_id,
                'name': os.path.basename(html_file),
                'label': os.path.basename(html_file),
                'description': 'HTML summary report for run_batch_model App'}

    def _generate_html_report(self):
        # Get the workspace name from the parameters
        ws_name = self.params["workspace"]

        # Visualize the result in html
        html_report_viz_file = self.visualize_hdf_in_html()

        self.html_files.append(html_report_viz_file)

        # Save the html to the report dictionary
        report_params = {
            # message is an optional field.
            # A string that appears in the summary section of the result page
            'message': "Say something...",

            # A list of typed objects created during the execution
            #   of the App. This can only be used to refer to typed
            #   objects in the workspace and is separate from any files
            #   generated by the app.
            # See a working example here:
            #   https://github.com/kbaseapps/kb_deseq/blob/586714d/lib/kb_deseq/Utils/DESeqUtil.py#L262-L264
            # 'objects_created': objects_created_in_app,

            # A list of strings that can be used to alert the user
            # 'warnings': warnings_in_app,

            # The workspace name or ID is included in every report
            'workspace_name': ws_name,

            # A list of paths or Shock IDs pointing to
            #   a single flat file. They appear in Files section
            'file_links': self.output_files,

            # HTML files that appear in “Links”
            'html_links': self.html_files,
            'direct_html_link_index': 0,
            'html_window_height': 333,
        } # end of report_params

        # Make the client, generate the report

        kbase_report_client = KBaseReport(self.callback_url)
        output = kbase_report_client.create_extended_report(report_params)

        # Return references which will allow inline display of
        # the report in the Narrative
        report_output = {'report_name': output['name'],
                        'report_ref': output['ref']}
        
        return report_output

class ReactiveTransportSimulatorRun1DUtil:
    def __init__(self,params):
        self.params = params
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.dfu = DataFileUtil(self.callback_url)
        self.output_files = []
        self.html_files = []
        self.data_folder = os.path.abspath('./data/')
        self.shared_folder = params['shared_folder']
        self.scratch_folder = os.path.join(params['shared_folder'],"scratch")

    def run_1d_model(self):

        print('params:',self.params)
        try:
            os.mkdir(self.scratch_folder)
        except OSError:
            print ("Creation of the directory %s failed" % self.scratch_folder)
        else:
            print ("Successfully created the directory %s " % self.scratch_folder)

        # move file templates from data folder to scratch folder 
        pflotran_input_temp = os.path.join(self.data_folder,'column_template.in')
        pflotran_db_temp    = os.path.join(self.data_folder,'database_template.dat')
        pflotran_input      = os.path.join(self.scratch_folder,'column.in')
        pflotran_db         = os.path.join(self.scratch_folder,'database.dat')
        stoi_csv_fba        = os.path.join(self.scratch_folder,'rxn_fba.csv')
        cpd_csv_fba         = os.path.join(self.scratch_folder,'cpd_fba.csv')

        # read inputs
        print("Input FBA model: ",self.params['input_FBA_model'])
        dfu                 = DataFileUtil(self.callback_url)
        fba_model           = dfu.get_objects({'object_refs': [self.params['input_FBA_model']]})['data'][0]
        print("FBA model name :",fba_model['data']['name'])
        nrxn                = int(self.params['number_simulated_reactions'])
        velocity            = float(self.params['velocity'])
        length              = float(self.params['length'])
        ngrid               = int(self.params['number_grids'])
        tot_time            = float(self.params['simulation_time'])
        timestep            = float(self.params['snapshot_period'])
        temperature         = float(self.params['temperature'])

        # collect the compound info
        cpdid2formula = dict()
        df_cpd = pd.DataFrame({'formula':[None]})
        for compound in fba_model['data']['modelcompounds']:
            cpdid2formula[compound['id']] = compound['formula']
            if 'biom' in compound['id']:
                df_cpd = df_cpd.append({'formula':'BIOMASS'}, ignore_index=True)
            else:
                df_cpd = df_cpd.append({'formula':compound['formula']}, ignore_index=True)
        df_cpd.insert(len(df_cpd.columns),'initial_concentration(mol/L)',0.01,True)
        df_cpd.loc[df_cpd.formula == 'BIOMASS', 'initial_concentration(mol/L)'] = 0.001
        df_cpd.insert(len(df_cpd.columns),'inlet_concentration(mol/L)',1,True)
        df_cpd.loc[df_cpd.formula == 'BIOMASS', 'inlet_concentration(mol/L)'] = 0
        df_cpd['formula'].replace('', np.nan, inplace=True)
        df_cpd = df_cpd.dropna()
        df_cpd.to_csv(cpd_csv_fba,index=False)
        print("Compounds saved. \n")
        
        # collect donor, acceptor, biom from reactions
        """
            donor : "~/modelcompounds/id/xcpd2_c0"
            acceptor : "~/modelcompounds/id/acceptor_c0"
            biom : "~/modelcompounds/id/biom_c0"
        """
        
        rxn_ref = ['r'+str(i+1) for i in range(nrxn)]
        df_rxn = pd.DataFrame({'rxn_ref':rxn_ref,'rxn_id':None,'DOC_formula':None})

        # selected_reactions = random.choices(fba_model['data']['modelreactions'],k=nrxn)
        selected_reactions = []
        selected_cpd       = []
        i = 0
        while i < nrxn:
            irxn = random.choice(fba_model['data']['modelreactions'])
            acceptor_flag = False
            for reagent in irxn['modelReactionReagents']:
                cpdid = reagent['modelcompound_ref'].split('/id/')[1]
                if 'acceptor' in cpdid:
                    acceptor_flag = True
                if 'xcpd' in cpdid:
                    doc = cpdid2formula[cpdid]
                    selected_cpd.append(doc)      
            if acceptor_flag and selected_cpd.count(doc) == 1:    
                selected_reactions.append(irxn)
                i += 1

        for reaction_idx,reaction_val in enumerate(selected_reactions):
            df_rxn['rxn_id'].iloc[reaction_idx] = reaction_val['id']
            for reagent in reaction_val['modelReactionReagents']:
                cpdid = reagent['modelcompound_ref'].split('/id/')[1]
                formula = cpdid2formula[cpdid]
                coef    = reagent['coefficient']

                if "xcpd" in cpdid:
                    df_rxn['DOC_formula'].iloc[reaction_idx] = formula

                if "biom" in cpdid:
                    formula = 'BIOMASS'

                if not formula in df_rxn.columns:
                    temp = ['0']*df_rxn.shape[0]
                    df_rxn.insert(len(df_rxn.columns),formula,temp,True)
                    df_rxn[formula].iloc[reaction_idx] = coef
                else:
                    df_rxn[formula].iloc[reaction_idx] = coef

        print(df_rxn.columns)
        print(df_rxn.head())
        df_rxn.to_csv(stoi_csv_fba,index=False)
        print("Selected reactions saved. \n")


        # read initial and boundary conditions from /bin/module/data
        init_cond = cpd_csv_fba

        # generate sandbox file
        sb_file = os.path.join(self.scratch_folder,'reaction_sandbox_pnnl_cyber.F90')
        var = ['mu_max','vh','k_deg','cc','activation_energy','reference_temperature']
        var_unit = ['1/sec','m^3','1/sec','M','J/mol','K']
        generate_sandbox_code(nrxn,var,var_unit,sb_file,stoi_csv_fba)
        print("Sandbox file generated.")

        # format sandbox fortran code
        fmt_sb_cmd = 'fprettify ' + sb_file
        process = subprocess.Popen(fmt_sb_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Sandbox file formatted.")

        # copy sandbox file to src dir and recompile pflotran
        src_dir = '/bin/pflotran/src/pflotran'
        copy(sb_file,src_dir)
        print(os.getcwd())
        compile_pflotran_cmd = 'sh ./data/compile.sh'
        process = subprocess.Popen(compile_pflotran_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Compile PFLOTRAN output:",output[-300:])
        print("Complile PFLOTRAN err:",error)
        pprint(os.listdir(self.scratch_folder))

        # generate 1d input deck
        self.generate_pflotran_input_1d(pflotran_input_temp,stoi_csv_fba,cpd_csv_fba,
        pflotran_input,velocity,length,ngrid,tot_time,timestep,temperature)
        print("Batch input deck generated.")

        # generate database 
        update_pflotran_database(stoi_csv_fba,pflotran_db_temp,pflotran_db)
        print("Database generated.")

        # running pflotran
        exepath = '/bin/pflotran/src/pflotran/pflotran'
        run_pflotran_cmd = exepath + ' -n 1 -pflotranin ' + pflotran_input
        process = subprocess.Popen(run_pflotran_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("Running PFLOTRAN output:",output[-300:])
        print("Running PFLOTRAN err:",error)
        pprint(os.listdir(self.scratch_folder))

        h5_file = os.path.join(self.scratch_folder,'column.h5')
        if os.path.isfile(h5_file):
            print ("Successfully run PFLOTRAN")
        else:
            print ("Fail to run PFLOTRAN")

        # generate plots in /kb/module/work/tmp/scratch/
        # self.plot_time_series_batch(h5_file)
        
        
        # Attach output
        self.output_files.append(
            {'path': cpd_csv_fba,
             'name': os.path.basename(cpd_csv_fba),
             'label': os.path.basename(cpd_csv_fba),
             'description': 'compounds'}
        ) 
        self.output_files.append(
            {'path': stoi_csv_fba,
             'name': os.path.basename(stoi_csv_fba),
             'label': os.path.basename(stoi_csv_fba),
             'description': 'reactions stoichiometry table'}
        )      
        self.output_files.append(
            {'path': sb_file,
             'name': os.path.basename(sb_file),
             'label': os.path.basename(sb_file),
             'description': 'Sandbox source code'}
        )              
        self.output_files.append(
            {'path': pflotran_input,
             'name': os.path.basename(pflotran_input),
             'label': os.path.basename(pflotran_input),
             'description': '1d column reaction input deck for PFLOTRAN'}
        )  
        self.output_files.append(
            {'path': pflotran_db,
             'name': os.path.basename(pflotran_db),
             'label': os.path.basename(pflotran_db),
             'description': '1d column reaction input deck for PFLOTRAN'}
        )  
        self.output_files.append(
            {'path': h5_file,
             'name': os.path.basename(h5_file),
             'label': os.path.basename(h5_file),
             'description': 'H5 file generated by PFLOTRAN 1d column reaction'}
        )
        # fig_name = 'time_series_plot.png'
        # fig_file = os.path.join(self.scratch_folder,fig_name) 
        # self.output_files.append(
        #     {'path': fig_file,
        #      'name': os.path.basename(fig_file),
        #      'label': os.path.basename(fig_file),
        #      'description': 'Plots of breakthrough curves generated by PFLOTRAN batch reaction'}
        # )

        # Return the report
        return self._generate_html_report()

    def generate_pflotran_input_1d(self,template,stoi_file,icbc_file,output_file,
    velocity,length,ngrid,tot_time,timestep,temp):
        file = open(template,'r')
        rxn_df = pd.read_csv(stoi_file)
        init_df = pd.read_csv(icbc_file)

        primary_species_charge = []
        primary_species_nocharge = []
        for spec in list(rxn_df.columns):
            if spec in ['rxn_id','DOC_formula','rxn_ref','H2O','BIOMASS']:
                continue
            primary_species_nocharge.append(spec)
            if spec=='NH4':
                primary_species_charge.append('NH4+')
                continue
            if spec=='HCO3':
                primary_species_charge.append('HCO3-')
                continue
            if spec=='H':
                primary_species_charge.append('H+')
                continue
            if spec=='HS':
                primary_species_charge.append('HS-')
                continue
            if spec=='HPO4':
                primary_species_charge.append('HPO4-')  
                continue
            primary_species_charge.append(spec) 

        init_cond   = [init_df.loc[init_df['formula']==i,'initial_concentration(mol/L)'].iloc[0] for i in primary_species_nocharge]
        init_biom   = init_df.loc[init_df['formula']=='BIOMASS','initial_concentration(mol/L)'].iloc[0]
        inlet_cond  = [init_df.loc[init_df['formula']==i,'inlet_concentration(mol/L)'].iloc[0] for i in primary_species_nocharge]
        inlet_biom  = init_df.loc[init_df['formula']=='BIOMASS','inlet_concentration(mol/L)'].iloc[0]
        for idx,val in enumerate(primary_species_nocharge):
            print("The initial concentration of {} is {} mol/L \n".format(val,init_cond[idx]))
            print("The inlet concentration of {} is {} mol/L \n".format(val,inlet_cond[idx]))

        pri_spec = ""
        pri_spec_init = ""
        new_file_content = ""
        for line in file:     
            if 'DATASET' in line:
                new_file_content += '  DATASET {} 0 0 m/h'.format(velocity) + "\n"

            elif 'NXYZ' in line:
                new_file_content += '  NXYZ {} 1 1'.format(ngrid) + "\n"

            elif 'PRIMARY_SPECIES' in line:
                new_file_content += line
                for i in primary_species_charge:
                    pri_spec += "    " + i + "\n"  
                new_file_content += "    " + pri_spec + "\n" 

            elif 'BOUNDS' in line:
                new_file_content += line
                new_file_content += "    0.d0 -1.d20 -1.d20" + "\n"
                new_file_content += "    {}   1.d20 1.d20".format(length) + "\n"

            elif 'REGION outlet' in line:
                new_file_content += line
                new_file_content += "  COORDINATES" + "\n"
                new_file_content += "    {} -1.d20 -1.d20".format(length) + "\n"
                new_file_content += "    {} -1.d20 -1.d20".format(length) + "\n"
                new_file_content += "  /" + "\n"
                new_file_content += "  FACE EAST" + "\n"

            elif 'CONSTRAINT initial' in line:
                new_file_content += line
                new_file_content += "  CONCENTRATIONS" + "\n"
                for j in range(len(primary_species_charge)):
                    new_file_content += "    {}        {} T".format(primary_species_charge[j],init_cond[j])+ "\n"
                new_file_content += "  /" + "\n"
                new_file_content += "  IMMOBILE" + "\n"
                new_file_content += "    BIOMASS        {} ".format(init_biom) + "\n"
                new_file_content += "  /"   

            elif 'CONSTRAINT inlet' in line:
                new_file_content += line
                new_file_content += "  CONCENTRATIONS" + "\n"
                for j in range(len(primary_species_charge)):
                    new_file_content += "    {}        {} T".format(primary_species_charge[j],inlet_cond[j])+ "\n"
                new_file_content += "  /" + "\n"
                new_file_content += "  IMMOBILE" + "\n"
                new_file_content += "    BIOMASS        {} ".format(inlet_biom) + "\n"
                new_file_content += "  /"   

            elif 'FINAL_TIME' in line:
                new_file_content += "  FINAL_TIME {} h".format(tot_time) + "\n"
                
            elif 'FINAL_TIME' in line:
                new_file_content += "  FINAL_TIME {} h".format(tot_time) + "\n"
                
            elif 'MAXIMUM_TIMESTEP_SIZE' in line:
                new_file_content += "  MAXIMUM_TIMESTEP_SIZE {} h".format(timestep) + "\n"
                
            elif 'PERIODIC TIME' in line:
                new_file_content += "    PERIODIC TIME {} h".format(timestep) + "\n"        
                
            elif 'REFERENCE_TEMPERATURE' in line:
                new_file_content += "      REFERENCE_TEMPERATURE {} ! degrees C".format(temp) + "\n"
                
            else:
                new_file_content += line  
                
        writing_file = open(output_file, "w")
        writing_file.write(new_file_content)
        writing_file.close()
        print('The batch input deck is updated.')
        return

    def plot_time_series_batch(self,h5_file):
        obs_coord = [0.5,0.5,0.5]

        file = h5py.File(h5_file,'r+')
        time_str = [list(file.keys())[i] for i in range(len(list(file.keys()))) if list(file.keys())[i][0:4] == "Time"]
        time_unit = time_str[0][-1]
        time = sorted([float(time_str[i].split()[1]) for i in range(len(time_str))])
        bound = []
        bound.append(file['Coordinates']['X [m]'][0])
        bound.append(file['Coordinates']['X [m]'][-1])
        bound.append(file['Coordinates']['Y [m]'][0])
        bound.append(file['Coordinates']['Y [m]'][-1])
        bound.append(file['Coordinates']['Z [m]'][0])
        bound.append(file['Coordinates']['Z [m]'][-1])
        nxyz = []
        nxyz.append(len(file['Coordinates']['X [m]'])-1)
        nxyz.append(len(file['Coordinates']['Y [m]'])-1)
        nxyz.append(len(file['Coordinates']['Z [m]'])-1)

        x_coord = (np.linspace(bound[0],bound[1],nxyz[0]+1)[:-1]+np.linspace(bound[0],bound[1],nxyz[0]+1)[1:])/2
        y_coord = (np.linspace(bound[2],bound[3],nxyz[1]+1)[:-1]+np.linspace(bound[2],bound[3],nxyz[1]+1)[1:])/2
        z_coord = (np.linspace(bound[4],bound[5],nxyz[2]+1)[:-1]+np.linspace(bound[4],bound[5],nxyz[2]+1)[1:])/2
        x_idx = np.argmin(np.absolute(x_coord-obs_coord[0]))
        y_idx = np.argmin(np.absolute(y_coord-obs_coord[1]))
        z_idx = np.argmin(np.absolute(z_coord-obs_coord[2]))
        time_zero = "Time:"+str(" %12.5E" % 0)+str(" %s" % time_unit)
        var_name = [x for x in list(file[time_zero].keys()) if 'Total' in x]
        var_value = np.zeros((len(var_name),len(time)))
        for i, itime in enumerate(time):
            time_slice = "Time:"+str(" %12.5E" % itime)+str(" %s" % time_unit)
        #     print(file[time_slice][var_name].keys())
            for j in range(len(var_name)):
                var_value[j,i] = file[time_slice][var_name[j]][x_idx][y_idx][z_idx]

        fig = plt.figure(num=1,dpi=150)
        first_doc = True
        for i in range(len(var_name)):
            if var_name[i][6] == 'C':
                if first_doc == True:
                    plt.plot(time,var_value[i,:],label='DOCs',color='k')[0]
                    first_doc = False
                else:
                    plt.plot(time,var_value[i,:],color='k')[0]
            else:
                plt.plot(time,var_value[i,:],label=var_name[i])[0]
            plt.ioff()

        plt.xlabel("Time (%s)" %time_unit)
        ylabel = 'Concentration [M]'
        plt.ylabel(ylabel)
        plt.legend(frameon=False,loc='upper center', bbox_to_anchor=(0.5, -0.15),ncol=3)
        fig_name = 'time_series_plot.png'
        fig_path = os.path.join(self.scratch_folder,fig_name)    
        plt.savefig(fig_path,dpi=150,bbox_inches='tight') 

        if os.path.isfile(fig_path):
            print ("Successfully generated time series plot")
        else:
            print ("Fail to generate time series plot")

        return

    def visualize_hdf_in_html(self):
        output_directory = os.path.join(self.shared_folder,'output') 
        os.makedirs(output_directory)
        print("output dir:", output_directory)
        html_file = os.path.join(output_directory,'summary.html')
        fig_name = 'time_series_plot.png'
        pflotran_out_name = 'batch.out'
        fig_path = os.path.join(self.scratch_folder,fig_name)
        pflotran_out_path = os.path.join(self.scratch_folder,pflotran_out_name)
        if os.path.isfile(fig_path):
            print ("Time series plot exists")
        else:
            print ("Time series plot does not exist")
        print("figpath:",fig_path)
        if os.path.isfile(pflotran_out_path):
            print ("PFLOTRAN output exists")
        else:
            print ("PFLOTRAN output does not exist")
        print("figpath:",pflotran_out_path)

        # copy(fig_path,'/kb/module/work/tmp/output')
        # copy(pflotran_out_path,'/kb/module/work/tmp/output')
        # with open(html_file, 'w') as f:
        #     f.write("""
        #         <!DOCTYPE html>
        #         <html>
        #         <body>

        #         <h1>PFLOTRAN-KBbase</h1>
        #         <p>PFLOTRAN output</p>
        #         <embed src="batch.out" width="480" height="960">
        #         <p>Visulize PFLOTRAN output</p>
        #         <img src="{}" alt="Time series plot" height="360" width="480"></img>
        #         </body>
        #         </html>
        #     """.format(fig_name))

        # test
        with open(html_file, 'w') as f:
            f.write("""
                <!DOCTYPE html>
                <html>
                <body>

                <h1>PFLOTRAN-KBbase</h1>
                <p>PFLOTRAN output</p>
                <embed src="batch.out" width="480" height="960">
                <p>Visulize PFLOTRAN output</p>
                <img src="" alt="Time series plot" height="360" width="480"></img>
                </body>
                </html>
            """)

        with open(html_file, 'r') as f:
            print("html_file:",f.readlines())

        report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,
                                                  'pack': 'zip'})['shock_id']

        return {'shock_id': report_shock_id,
                'name': os.path.basename(html_file),
                'label': os.path.basename(html_file),
                'description': 'HTML summary report for run_batch_model App'}

    def _generate_html_report(self):
        # Get the workspace name from the parameters
        ws_name = self.params["workspace"]

        # Visualize the result in html
        html_report_viz_file = self.visualize_hdf_in_html()

        self.html_files.append(html_report_viz_file)

        # Save the html to the report dictionary
        report_params = {
            # message is an optional field.
            # A string that appears in the summary section of the result page
            'message': "Say something...",

            # A list of typed objects created during the execution
            #   of the App. This can only be used to refer to typed
            #   objects in the workspace and is separate from any files
            #   generated by the app.
            # See a working example here:
            #   https://github.com/kbaseapps/kb_deseq/blob/586714d/lib/kb_deseq/Utils/DESeqUtil.py#L262-L264
            # 'objects_created': objects_created_in_app,

            # A list of strings that can be used to alert the user
            # 'warnings': warnings_in_app,

            # The workspace name or ID is included in every report
            'workspace_name': ws_name,

            # A list of paths or Shock IDs pointing to
            #   a single flat file. They appear in Files section
            'file_links': self.output_files,

            # HTML files that appear in “Links”
            'html_links': self.html_files,
            'direct_html_link_index': 0,
            'html_window_height': 333,
        } # end of report_params

        # Make the client, generate the report

        kbase_report_client = KBaseReport(self.callback_url)
        output = kbase_report_client.create_extended_report(report_params)

        # Return references which will allow inline display of
        # the report in the Narrative
        report_output = {'report_name': output['name'],
                        'report_ref': output['ref']}
        
        return report_output

def generate_sandbox_code(nrxn,var,var_unit,sb_file,stoi_file):
    rxn_name = 'cyber'
    rxn_df = pd.read_csv(stoi_file)
    primary_species_charge = []
    primary_species_nocharge = []
    for spec in list(rxn_df.columns):
        if spec in ['rxn_id','DOC_formula','rxn_ref','H2O']:
            continue
        primary_species_nocharge.append(spec)
        if spec=='NH4':
            primary_species_charge.append('NH4+')
            continue
        if spec=='HCO3':
            primary_species_charge.append('HCO3-')
            continue
        if spec=='H':
            primary_species_charge.append('H+')
            continue
        if spec=='HS':
            primary_species_charge.append('HS-')
            continue
        if spec=='HPO4':
            primary_species_charge.append('HPO4-')
            continue
        primary_species_charge.append(spec)

    sandbox_file = open(sb_file,'w+')
    sb = '''
module Reaction_Sandbox_{}_class

    use Reaction_Sandbox_Base_class

    use Global_Aux_module
    use Reactive_Transport_Aux_module

    use PFLOTRAN_Constants_module

    implicit none

    private

#include "petsc/finclude/petscsys.h"
'''
    sb = sb.format(rxn_name.capitalize())

    for idx,item in enumerate(primary_species_nocharge):
        sb = sb+"  PetscInt, parameter :: {}_MASS_STORAGE_INDEX = {}\n".format(item,idx+1)

    sb = sb+'''
    type, public, &
    extends(reaction_sandbox_base_type) :: reaction_sandbox_{}_type
'''.format(rxn_name)

    for idx,item in enumerate(primary_species_nocharge):
        sb = sb+"    PetscInt :: {}_id \n".format(item.lower())

    for i in var:
        sb = sb+"    PetscReal :: {} \n".format(i)


    sb = sb+'''
    PetscReal :: nrxn
    PetscBool :: store_cumulative_mass
    PetscInt :: offset_auxiliary
    contains
    procedure, public :: ReadInput => {}Read
    procedure, public :: Setup => {}Setup
    procedure, public :: Evaluate => {}React
    procedure, public :: Destroy => {}Destroy
    end type reaction_sandbox_{}_type

    public :: {}Create

contains

! ************************************************************************** !
'''.format(rxn_name.capitalize(),rxn_name.capitalize(),rxn_name.capitalize(),rxn_name.capitalize(),
            rxn_name,rxn_name.capitalize())

#----------------------------------------------------------------------------
#
#                            function create()
#
#----------------------------------------------------------------------------
    sb = sb+'''
function {}Create()
#include "petsc/finclude/petscsys.h"
    use petscsys
    implicit none

    class(reaction_sandbox_{}_type), pointer :: {}Create

    allocate({}Create)
'''.format(rxn_name.capitalize(),rxn_name,rxn_name.capitalize(),rxn_name.capitalize())

    for i in primary_species_nocharge:
        sb = sb+"  {}Create%{}_id = UNINITIALIZED_INTEGER \n".format(rxn_name.capitalize(),i.lower())

    for i in var:
        if i.lower() == 'reference_temperature':
            sb = sb + '  CyberCreate%reference_temperature = 298.15d0 ! 25 C\n'
        else:
            sb = sb+"  {}Create%{} = UNINITIALIZED_DOUBLE \n".format(rxn_name.capitalize(),i)

    sb = sb+'''
    {}Create%nrxn = UNINITIALIZED_INTEGER
    {}Create%store_cumulative_mass = PETSC_FALSE

    nullify({}Create%next)
    print *, '{}Creat Done'
end function {}Create

! ************************************************************************** !
'''.format(rxn_name.capitalize(),rxn_name.capitalize(),rxn_name.capitalize(),
rxn_name.capitalize(),rxn_name.capitalize())

#----------------------------------------------------------------------------
#
#                            function read()
#
#----------------------------------------------------------------------------
    sb = sb+'''
! ************************************************************************** !

subroutine {}Read(this,input,option)

    use Option_module
    use String_module
    use Input_Aux_module

    implicit none

    class(reaction_sandbox_{}_type) :: this
    type(input_type), pointer :: input
    type(option_type) :: option

    PetscInt :: i
    character(len=MAXWORDLENGTH) :: word, internal_units, units
    character(len=MAXSTRINGLENGTH) :: error_string

    error_string = 'CHEMISTRY,REACTION_SANDBOX,{}'
    call InputPushBlock(input,option)
    do
    call InputReadPflotranString(input,option)
    if (InputError(input)) exit
    if (InputCheckExit(input,option)) exit

    call InputReadCard(input,option,word)
    call InputErrorMsg(input,option,'keyword',error_string)
    call StringToUpper(word)

    select case(trim(word))
'''.format(rxn_name.capitalize(),rxn_name.lower(),rxn_name.upper())

    for idx,item in enumerate(var):
        if item!='reference_temperature':
            sb = sb+'''
        case('{}')
        call InputReadDouble(input,option,this%{})
        call InputErrorMsg(input,option,'{}',error_string)
        call InputReadAndConvertUnits(input,this%{},'{}', &
                                        trim(error_string)//',{}',option)
        '''.format(item.upper(),item.lower(),item.lower(),item.lower(),
                    var_unit[idx],item.lower())
        else:
            sb = sb+'''
        case('REFERENCE_TEMPERATURE')
        call InputReadDouble(input,option,this%reference_temperature)
        call InputErrorMsg(input,option,'reference temperature [C]', &
                            error_string)
        this%reference_temperature = this%reference_temperature + 273.15d0
        '''
    sb = sb+'''
        case default
        call InputKeywordUnrecognized(input,word,error_string,option)
    end select
    enddo
    call InputPopBlock(input,option)
end subroutine {}Read

! ************************************************************************** !
'''.format(rxn_name.capitalize())

#----------------------------------------------------------------------------
#
#                            function setup()
#
#----------------------------------------------------------------------------
    sb = sb+'''
subroutine {}Setup(this,reaction,option)

    use Reaction_Aux_module, only : reaction_rt_type, GetPrimarySpeciesIDFromName
    use Reaction_Immobile_Aux_module, only : GetImmobileSpeciesIDFromName
    use Reaction_Mineral_Aux_module, only : GetKineticMineralIDFromName
    use Option_module

    implicit none

    class(reaction_sandbox_{}_type) :: this
    class(reaction_rt_type) :: reaction
    type(option_type) :: option

    character(len=MAXWORDLENGTH) :: word
    PetscInt :: irxn

    PetscReal, parameter :: per_day_to_per_sec = 1.d0 / 24.d0 / 3600.d0
'''.format(rxn_name.capitalize(),rxn_name.lower())

    for idx,item in enumerate(primary_species_charge):
        if item.upper()!='BIOMASS':
            sb = sb+'''
    word = '{}'
    this%{}_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    '''.format(item.upper(),primary_species_nocharge[idx].lower())
        else:
            sb = sb+'''
    word = 'BIOMASS'
    this%biomass_id = &
    GetImmobileSpeciesIDFromName(word,reaction%immobile,option) + reaction%offset_immobile
        '''

    sb = sb+'''
    if (this%store_cumulative_mass) then
    this%offset_auxiliary = reaction%nauxiliary
    reaction%nauxiliary = reaction%nauxiliary + {}
    endif

end subroutine {}Setup

! ************************************************************************** !
'''.format(len(primary_species_charge)*2,rxn_name.capitalize())

#----------------------------------------------------------------------------
#
#                            function PlotVariables()
#
#----------------------------------------------------------------------------
    sb = sb+'''
subroutine {}AuxiliaryPlotVariables(this,list,reaction,option)

    use Option_module
    use Reaction_Aux_module
    use Output_Aux_module
    use Variables_module, only : REACTION_AUXILIARY

    implicit none

    class(reaction_sandbox_{}_type) :: this
    type(output_variable_list_type), pointer :: list
    type(option_type) :: option
    class(reaction_rt_type) :: reaction

    character(len=MAXWORDLENGTH) :: names({})
    character(len=MAXWORDLENGTH) :: word
    character(len=MAXWORDLENGTH) :: units
    PetscInt :: indices({})
    PetscInt :: i

'''.format(rxn_name.capitalize(),rxn_name.lower(),len(primary_species_charge),len(primary_species_charge))

    for idx,item in enumerate(primary_species_charge):
        sb = sb+"  names({}) = '{}'\n".format(idx+1,item.upper())

    for idx,item in enumerate(primary_species_nocharge):
        sb = sb+"  indices({}) = {}_MASS_STORAGE_INDEX\n".format(idx+1,item.upper())

    sb = sb+'''
    if (this%store_cumulative_mass) then
    do i = 1, {}
        word = trim(names(i)) // ' Rate'
        units = 'mol/m^3-sec'
        call OutputVariableAddToList(list,word,OUTPUT_RATE,units, &
                                    REACTION_AUXILIARY, &
                                    this%offset_auxiliary+indices(i))
    enddo
    do i = 1, {}
        word = trim(names(i)) // ' Cum. Mass'
        units = 'mol/m^3'
        call OutputVariableAddToList(list,word,OUTPUT_GENERIC,units, &
                                    REACTION_AUXILIARY, &
                                    this%offset_auxiliary+{}+indices(i))
    enddo
    endif

end subroutine {}AuxiliaryPlotVariables

! ************************************************************************** !
'''.format(len(primary_species_charge),len(primary_species_charge),len(primary_species_charge),rxn_name.capitalize())

#----------------------------------------------------------------------------
#
#                            function react()
#
#----------------------------------------------------------------------------
    sb = sb+'''
subroutine {}React(this,Residual,Jacobian,compute_derivative, &
                            rt_auxvar,global_auxvar,material_auxvar,reaction, &
                            option)

    use Option_module
    use Reaction_Aux_module
    use Material_Aux_class

    implicit none

    class(reaction_sandbox_{}_type) :: this
    type(option_type) :: option
    class(reaction_rt_type) :: reaction
    ! the following arrays must be declared after reaction
    PetscReal :: Residual(reaction%ncomp)
    PetscReal :: Jacobian(reaction%ncomp,reaction%ncomp)
    type(reactive_transport_auxvar_type) :: rt_auxvar
    type(global_auxvar_type) :: global_auxvar
    class(material_auxvar_type) :: material_auxvar

    PetscInt, parameter :: iphase = 1
    PetscReal :: L_water
    PetscReal :: kg_water

    PetscInt :: i, j, irxn
'''.format(rxn_name.capitalize(),rxn_name.lower())

    for idx, item in enumerate(primary_species_nocharge):
        sb = sb+"  PetscReal :: C_{} \n".format(item.lower()) 

    for i in range(nrxn):
        sb = sb+"  PetscReal :: r{}doc,r{}o2 \n".format(i+1,i+1)

    for i in range(nrxn):
        sb = sb+"  PetscReal :: r{}kin \n".format(i+1)

    sb = sb+"  PetscReal :: sumkin \n"
    for i in range(nrxn):
        sb = sb+"  PetscReal :: u{} \n".format(i+1)

    sb = sb+"  PetscReal :: molality_to_molarity\n  PetscReal :: temperature_scaling_factor\n  PetscReal :: mu_max_scaled\n"
    for i in range(nrxn):
        sb = sb+"  PetscReal :: k{}_scaled \n".format(i+1)

    sb = sb+"  PetscReal :: k_deg_scaled"

    sb = sb+'''
    PetscReal :: volume, rate_scale
    PetscBool :: compute_derivative

    PetscReal :: rate({})

    volume = material_auxvar%volume
    L_water = material_auxvar%porosity*global_auxvar%sat(iphase)* &
            volume*1.d3 ! m^3 -> L
    kg_water = material_auxvar%porosity*global_auxvar%sat(iphase)* &
                global_auxvar%den_kg(iphase)*volume

    molality_to_molarity = global_auxvar%den_kg(iphase)*1.d-3

    if (reaction%act_coef_update_frequency /= ACT_COEF_FREQUENCY_OFF) then
    option%io_buffer = 'Activity coefficients not currently supported in &
        &{}React().'
    call printErrMsg(option)
    endif

    temperature_scaling_factor = 1.d0
    if (Initialized(this%activation_energy)) then
    temperature_scaling_factor = &
        exp(this%activation_energy/IDEAL_GAS_CONSTANT* &
            (1.d0/this%reference_temperature-1.d0/(global_auxvar%temp+273.15d0)))
    endif

'''.format(nrxn,rxn_name.capitalize())

    sb = sb +"  ! concentrations are molarities [M]"
    for i in primary_species_nocharge:
        if i.upper()!='BIOMASS':
            sb = sb+'''
    C_{} = rt_auxvar%pri_molal(this%{}_id)* &
        rt_auxvar%pri_act_coef(this%{}_id)*molality_to_molarity
        '''.format(i.lower(),i.lower(),i.lower())
        else:
            sb = sb+'''
    C_biomass = rt_auxvar%immobile(this%biomass_id-reaction%offset_immobile)
        '''
    sb = sb +'''
    mu_max_scaled = this%mu_max * temperature_scaling_factor
    k_deg_scaled = this%k_deg * temperature_scaling_factor
'''

    sb = sb+generate_rate_expression(primary_species_nocharge, stoi_file, rxn_name)

    sb = sb+'''
end subroutine {}React


! ************************************************************************** !

subroutine {}Destroy(this)
    use Utility_module

    implicit none

    class(reaction_sandbox_{}_type) :: this

    print *, '{}Destroy Done'

end subroutine {}Destroy

end module Reaction_Sandbox_{}_class
'''.format(rxn_name.capitalize(),rxn_name.capitalize(),rxn_name.lower(),
            rxn_name.capitalize(),rxn_name.capitalize(),rxn_name.capitalize())
    sandbox_file.write(sb)

    print('Sandbox code is generated at {}.'.format(sb_file))
    return


def generate_rate_expression(primary_species_nocharge, stoi_file, rxn_name):
    rxn_df = pd.read_csv(stoi_file)
    rxn_df = rxn_df.set_index('rxn_ref')

    rkin = {}
    for i in range(len(rxn_df)):
#         doc_name = rxn_df.iloc[i,0]
#         doc_name = re.sub('[-+)]','',doc_name)
        doc_name = rxn_df['DOC_formula'].iloc[i] 
        doc_name = doc_name.lower()
        print(doc_name)
        doc_sto = rxn_df[rxn_df['DOC_formula'].loc['r'+str(i+1)]].loc['r'+str(i+1)]
        o2_sto = rxn_df['O2'].loc['r'+str(i+1)]
        rdoc_i = '  r'+str(i+1)+'doc = '+'exp('+str(doc_sto)+'/(this%vh * C_' + doc_name+'))'
        ro2_i = '  r'+str(i+1)+'o2 = '+'exp('+str(o2_sto)+'/(this%vh * C_o2))'
        rkin_i = '  r'+str(i+1)+'kin = ' + 'mu_max_scaled * '+'r'+str(i+1)+'doc'+' * ' + 'r'+str(i+1)+'o2'
        rkin[doc_name] = [rdoc_i,ro2_i,rkin_i]

    sumkin = '  sumkin = '
    for i in range(len(rxn_df)):
        if i == len(rxn_df)-1:
            sumkin = sumkin + '         r' + str(i+1) + 'kin '
        elif i == 0:
            sumkin = sumkin + 'r' + str(i+1) + 'kin + & \n'
        else:
            sumkin = sumkin + '         r' + str(i+1) + 'kin + & \n'

    u = []
    for i in range(len(rxn_df)):
        u.append('  u' + str(i+1) + ' = 0.d0')
        u.append('  if (r' + str(i+1) + 'kin > 0.d0) u' + str(i+1) + ' = r' + str(i+1) + 'kin/sumkin' )

    rate = []
    for i in range(len(rxn_df)):
        rate.append('  rate(' + str(i+1) + ') = u' + str(i+1) + '*r' + str(i+1) + 'kin*(1-C_biomass/this%cc)')

    res = {}
    for i in primary_species_nocharge:
        icol = rxn_df.columns.get_loc(i)
        i = i.lower()
        i_id = 'this%'+i+'_id'

        res_i = ['  Residual(' + i_id + ') = Residual(' + i_id +')  &']
        space_idx = res_i[0].find('=')
#         first_rate_flag = True
        for irow in range(len(rxn_df)):
            if pd.isnull(rxn_df.iloc[irow,icol]):
                continue
            sto_i = str(rxn_df.iloc[irow,icol])
            if sto_i[0] == '-':
#                 if first_rate_flag:
#                     res_i[0] = re.sub('[-]','+',res_i[0])
#                     first_rate_flag = False
                sto_i = re.sub('[-]','',sto_i)
                res_i_temp = ' '*space_idx + ' + ' + str(sto_i) + ' * rate(' + str(irow+1) +') * C_biomass * L_water &'
            else:
                res_i_temp = ' '*space_idx + ' - ' + str(sto_i) + ' * rate(' + str(irow+1) +') * C_biomass * L_water &'

            res_i.append(res_i_temp)

        res_i[-1] = res_i[-1][0:-2]
        res_i[-1] = res_i[-1]
        res[i_id] = res_i

    res['this%biomass_id'].append('  Residual(this%biomass_id) = Residual(this%biomass_id) + k_deg_scaled * C_biomass * L_water \n')


    mass = {}
    for i in primary_species_nocharge:
        icol = rxn_df.columns.get_loc(i)
        i = i.lower()
        i_id = i.upper()

        mass_i = ['        i = this%offset_auxiliary + ' + i_id + '_MASS_STORAGE_INDEX' ]
        mass_i.append('        rt_auxvar%auxiliary_data(i) = &')
        space_idx = mass_i[0].find('_')
        for irow in range(len(rxn_df)):
            if pd.isnull(rxn_df.iloc[irow,icol]):
                continue

            sto_i = str(rxn_df.iloc[irow,icol])
            if sto_i[0] == '-':
                sto_i = re.sub('[-]','',sto_i)
                mass_i_temp = ' '*space_idx + ' + ' + str(sto_i) + ' * rate(' + str(irow+1) +') * rate_scale &'
            else:
                mass_i_temp = ' '*space_idx + ' + ' + str(sto_i) + ' * rate(' + str(irow+1) +') * rate_scale &'

            mass_i.append(mass_i_temp)

        mass_i[-1] = mass_i[-1][0:-2]
        mass_i[-1] = mass_i[-1]
        mass[i_id] = mass_i

    rate_expr = '\n'
    for key, values in rkin.items():
        for i in range(len(values)):
            rate_expr = rate_expr+values[i]+'\n'

    rate_expr = rate_expr+'\n'
    rate_expr = rate_expr+sumkin+'\n'

    rate_expr = rate_expr+'\n'
    for i in u:
        rate_expr = rate_expr+i+'\n'

    rate_expr = rate_expr+'\n'
    for i in rate:
        rate_expr = rate_expr+i+'\n'

    rate_expr = rate_expr+'\n'
    for key, values in res.items():
        for i in range(len(values)):
            rate_expr = rate_expr+values[i]+'\n'

    rate_expr = rate_expr+'''
    if (this%store_cumulative_mass) then
        rate_scale = C_biomass * L_water / volume
    '''
    for key, values in mass.items():
        for i in range(len(values)):
            rate_expr = rate_expr+values[i]+'\n'

    rate_expr = rate_expr+'''
    endif
    '''
    return rate_expr

def update_pflotran_database(stoi_file,dbase_temp_file,dbase_out_file):
    rxn_df = pd.read_csv(stoi_file)
    print(rxn_df['DOC_formula'].values)
    
    new_db_content = ""
    doc_db = ""
    file = open(dbase_temp_file,'r')
    for line in file:
        if "'C" not in line:
            new_db_content += line
        else:
            for i in rxn_df['DOC_formula'].values:
                doc_db += "'{}'".format(i)+" 3.0 0.0 100" + '\n'
            new_db_content += doc_db
    
    writing_file = open(dbase_out_file, "w")
    writing_file.write(new_db_content)
    writing_file.close()
    print('The database is updated.')
    return