# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
from ReactiveTransportSimulator.Utils.ReactiveTransportSimulatorUtil import ReactiveTransportSimulatorRunBatchUtil
from ReactiveTransportSimulator.Utils.ReactiveTransportSimulatorUtil import ReactiveTransportSimulatorRun1DUtil
from installed_clients.KBaseReportClient import KBaseReport
from installed_clients.DataFileUtilClient import DataFileUtil

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
    GIT_URL = "https://github.com/kchen8921/ReactiveTransportSimulator-KBase.git"
    GIT_COMMIT_HASH = "c0ca935bb20fa4b38f451e58aceecbb2aa662209"

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


    def run_batch_model(self, ctx, params):
        """
        Thi function enables users to run a pflotran batch model using FBA model and user-provided initial condition
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_batch_model
        params['shared_folder'] = self.shared_folder
        pu = ReactiveTransportSimulatorRunBatchUtil(params)
        output = pu.run_batch_model()
        #END run_batch_model

        # At some point might do deeper type checking...
        # if not isinstance(output, dict):
        #     raise ValueError('Method run_batch_model return value ' +
        #                      'output is not type dict as required.')
        # return the results
        return [output]

    def run_1d_model(self, ctx, params):
        """
        Thi function enables users to run a pflotran 1d column model using FBA model and user-provided initial and boundary conditions
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_batch_model
        params['shared_folder'] = self.shared_folder
        pu = ReactiveTransportSimulatorRun1DUtil(params)
        output = pu.run_1d_model()
        #END run_batch_model

        # At some point might do deeper type checking...
        # if not isinstance(output, dict):
        #     raise ValueError('Method run_1d_model return value ' +
        #                      'output is not type dict as required.')

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
