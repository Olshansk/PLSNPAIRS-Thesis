package pls.chrome.shared;

import pls.shared.StreamedProgressHelper;

public class ProgressDialogWatcher extends Thread {

	public StreamedProgressHelper progress = null;
	
	public void setProgressDialog(StreamedProgressHelper progress) {
		this.progress = progress;
	}
	
	public void run() {
		try {
			doTask();
		/*} catch(OutOfMemoryError em) {
			//progress.printError(em.getMessage());
			progress.printError("Out of memory error occured in the java heap space. You should allocate more RAM for plsnpairs. Try VM argument '-Xmx1g'. If you still receive this message try '-Xmx1500m' or '-Xmx2g', etc. \nRAM = random access memory; VM = virtual machine; g = gigabyte; m = megabyte; '-Xmx' argument can be set in the Eclipse run configuration window, or from command line.");
			progress.complete(); */
		} catch(Exception ex) {
			ex.printStackTrace();
			progress.printError(ex.getMessage());
			progress.complete();
		}
		
		//progress = null;
	}
	
	public void doTask() throws Exception {
		// Children should implement this, not run so that exceptions can be printed to progress dialog
	}
}
