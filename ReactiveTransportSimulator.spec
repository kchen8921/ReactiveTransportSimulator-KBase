/*
A KBase module: ReactiveTransportSimulator
*/

module ReactiveTransportSimulator {
    typedef structure {
        string report_name;
        string report_ref;
    } ReportResults;

    /*
        Thi function enables users to run a pflotran batch model with FBA model and initial conditions as inputs
    */
    funcdef run_batch_model(mapping<string,UnspecifiedObject> params) returns (ReportResults output) authentication required;

    /*
        Thi function enables users to run a pflotran 1D model with FBA model and initial conditions as inputs
    */
    funcdef run_1d_model(mapping<string,UnspecifiedObject> params) returns (ReportResults output) authentication required;
   
};
