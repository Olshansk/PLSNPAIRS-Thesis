package pls.chrome.shared;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JSeparator;
import javax.swing.JTabbedPane;
import javax.swing.JTable;

import pls.shared.fMRIContrastFileFilter;

import java.io.PrintStream;
import java.io.FileOutputStream;

import pls.shared.MLFuncs;

@SuppressWarnings("serial")
public class ContrastsFrame extends JFrame {
	
	public ContrastsFrame(Vector<String[]> sessionProfiles, 
						  Vector<Integer> conditionSelection,
						  String contrastFilePath) {
		
		super("New Contrasts");
		init(new ArrayList<String[]>(sessionProfiles), 
			 new ArrayList<Integer>(conditionSelection),
			 contrastFilePath);
		
	}
	
	public ContrastsFrame(Vector<String[]> sessionProfiles) {
		// TODO Auto-generated constructor stub for compatibility with
		// pls.chrome.analysis.PetAnalysisFrame.
	}
	
	private void init(ArrayList<String[]> sessionProfiles, 
					  ArrayList<Integer> conditionSelection,
					  String contrastFilePath) {
		
		setDefaultCloseOperation(DISPOSE_ON_CLOSE);
	    
		if(sessionProfiles.size() == 0) {
			JOptionPane.showMessageDialog(null, 
			"Group needs to be added before you can open contrasts window.", 
			"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
				
		if(conditionSelection.size() == 0){
			JOptionPane.showMessageDialog(null, 
			"Conditions must be selected before you can open the contrasts" +
			" window."	,"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		
		int greaterThanOne = 0;
		for(Integer c : conditionSelection){ 
			if(c == 1) greaterThanOne++;
			if(greaterThanOne > 1) break;
		}
		
		if (greaterThanOne < 2){
			JOptionPane.showMessageDialog(null, 
			"More than one condition must be selected before you can" +
			" open the contrasts window."
			,"Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
	    
		ContrastPanel contrasts = new ContrastPanel(sessionProfiles,
													conditionSelection,
													contrastFilePath);
	    
	    setJMenuBar(new ContrastsMenuBar(this, contrasts));
	    
	    add(contrasts);
	    
        // Display the window
        pack();
        
        // Position the frame on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        setVisible(true);
	}
}


final class ContrastsMenuBar extends BaseSaveMenuBar {
	
	private ContrastPanel contrasts = null;
	
	private ContrastTablePanel[] cTablePanels = null;
	
	private JTabbedPane tabs = null;
	
	public ContrastsMenuBar(JFrame frame, final ContrastPanel contrasts) {
		super(frame);
		
		this.cTablePanels = contrasts.cTablePanels;
		this.tabs = contrasts.tabs;
		this.contrasts = contrasts;
		
		setFileFilter(new fMRIContrastFileFilter(), "_fMRIcontrast.txt");
		JMenu fileMenu = getMenu(0);
		
        JMenuItem loadHelmert = new JMenuItem("Load Helmert Matrix");
        loadHelmert.setMnemonic('H');
        loadHelmert.getAccessibleContext().setAccessibleDescription("Load Helmert Matrix");
        loadHelmert.addActionListener(new ActionListener() { public void actionPerformed(ActionEvent e) { loadHelmert(); }});
        
        JMenuItem clear = new JMenuItem("Clear");
        clear.addActionListener(new ActionListener() { 
        	public void actionPerformed(ActionEvent e) { contrasts.clearTable(); }});
        
        
        fileMenu.add(new JSeparator(), 3);
        fileMenu.add(clear,4);
        fileMenu.add(loadHelmert, 5);
        fileMenu.add(new JSeparator(), 6);
        
        
        
	}
	
	public void loadHelmert() {
		JTable table = cTablePanels[tabs.getSelectedIndex()].table;
		double[][] helmert = MLFuncs.getRRIHelmertMatrix(table.getColumnCount());
		for(int i = 0; i < helmert.length; i++) {
			for(int j = 0; j < helmert[0].length; j++) {
				table.setValueAt(new Double(helmert[i][j]), i, j);
			}
		}
		// Reset rest of the table
		for(int i = helmert.length; i < table.getRowCount(); i++) {
			for(int j = 0; j < table.getColumnCount(); j++) {
				table.setValueAt("", i, j);
			}
		}
	}
	
	@Override
	public void save() {
		if(!contrasts.checkContrastAddsUpToZero()) return;
		PrintStream p = null;
		try {
			p = new PrintStream(new FileOutputStream(fileName));
		} catch(Exception ex) {
			JOptionPane.showMessageDialog(null, "Unable to save to " + fileName + ".", "Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
		for(int i = 0; i < cTablePanels.length; i++) {
			if(cTablePanels[i].table.getCellEditor() != null) {
				cTablePanels[i].table.getCellEditor().stopCellEditing();
			}
			for(int j = 0; j < cTablePanels[i].table.getColumnCount(); j++) {
				for(int k = 0; k < cTablePanels[i].table.getRowCount(); k++) {
					Object currValue = cTablePanels[i].table.getValueAt(k, j);
					if(currValue == null) {
						break;
					}
					p.print(currValue.toString() + "\t");
				}
				p.println();
			}
		}
		p.close();
	}
	
	@Override
	public void load() {
		contrasts.load(fileName);
	}
}