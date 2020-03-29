# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
from ReactiveTransportSimulator.Utils.ReactiveTransportSimulatorUtil import ReactiveTransportSimulatorRunBatchUtil
from installed_clients.KBaseReportClient import KBaseReport
#END_HEADER


class ReactiveTransportSimulator:
    '''
    Module Name:
    ReactiveTransportSimulator

    Module Description:
    A KBase module: ReactiveTransportSimulator
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = ""
    GIT_COMMIT_HASH = ""

    #BEGIN_CLASS_HEADER
    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']
        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        #END_CONSTRUCTOR
        pass


    # def run_ReactiveTransportSimulator(self, ctx, params):
    #     """
    #     This example function accepts any number of parameters and returns results in a KBaseReport
    #     :param params: instance of mapping from String to unspecified object
    #     :returns: instance of type "ReportResults" -> structure: parameter
    #        "report_name" of String, parameter "report_ref" of String
    #     """
    #     # ctx is the context object
    #     # return variables are: output
    #     #BEGIN run_ReactiveTransportSimulator
    #     report = KBaseReport(self.callback_url)
    #     report_info = report.create({'report': {'objects_created':[],
    #                                             'text_message': params['parameter_1']},
    #                                             'workspace_name': params['workspace_name']})
    #     output = {
    #         'report_name': report_info['name'],
    #         'report_ref': report_info['ref'],
    #     }
    #     #END run_ReactiveTransportSimulator

    #     # At some point might do deeper type checking...
    #     if not isinstance(output, dict):
    #         raise ValueError('Method run_ReactiveTransportSimulator return value ' +
    #                          'output is not type dict as required.')
    #     # return the results
    #     return [output]

    def run_batch_model(self, ctx, params):
        """
        Thi function enables users to run a pflotran batch simulation from an input plfotran model and fbamodel chemistry
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_PFLOTRAN
        params['shared_folder'] = self.shared_folder
        pu = ReactiveTransportSimulatorRunUtil(params)
        output = pu.run_batch_model() 
        # html_folder = os.path.join(self.shared_folder, 'html')
        # os.mkdir(html_folder)

        # html_str = "<html><head>KB-PFLOTRAN Report</head><body><br><br></body></html>"

        # with open(os.path.join(html_folder, "index.html"), 'w') as index_file:
        #     index_file.write(html_str)

        # report = KBaseReport(self.callback_url)
        # html_dir = {
        #     'path': html_folder,
        #     'name': 'index.html',  # MUST match the filename of your main html page
        #     'description': 'Thermo Stoich Wizard Report'
        # }
        # report_info = report.create_extended_report({
        #     'html_links': [html_dir],
        #     'direct_html_link_index': 0,
        #     'report_object_name': 'pflotran_report_' + uuid_string,
        #     'workspace_name': params['workspace_name']
        # })
        # # report_info = report.create({'report': {'objects_created':[],
        # #                                         'text_message': "OK"},
        # #                                         'workspace_name': params['workspace_name']})
        # output = {
        #     'report_name': report_info['name'],
        #     'report_ref': report_info['ref'],
        # }
        #END run_PFLOTRAN

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_PFLOTRAN return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
