package pls.chrome.sessionprofile;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.table.DefaultTableModel;

import nl.jj.swingx.gui.modal.JModalFrame;
import pls.chrome.shared.BaseMenuBar;

@SuppressWarnings("serial")
public class PetEditConditionsFrame extends JModalFrame {
	
	public PetEditConditionsFrame(PetSessionProfileFrame sessionFrame) {
		super("Edit Conditions");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);
	    
	    DefaultTableModel model = new DefaultTableModel();
	    JTable table = new JTable();
	    table.setModel(model);
	    table.setPreferredScrollableViewportSize(new Dimension(table.getPreferredScrollableViewportSize().width, 160));

	    model.addColumn("Condition Name");
	    /*******************************************
	    model.addColumn("Reference Scan Onset");
	    model.addColumn("Number of Reference Scans");
	    /********************************************/
	    int numRows = 0;
	    for(String[] conditionRow : sessionFrame.conditionInfo) {
	    	model.addRow(conditionRow);
	    	numRows++;
	    }
	    
	    for(; numRows < 100; numRows++) {
	    	model.addRow(new Object[]{});
	    }
	    
        JButton saveButton = new JButton("Store And Continue", new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/SendMail16.gif")));
        saveButton.setIconTextGap(15);
        saveButton.addActionListener(new PetConditionSaveListener(this, sessionFrame, table));

		setLayout(new BorderLayout());
		add(new JScrollPane(table), BorderLayout.NORTH);
		add(saveButton, BorderLayout.SOUTH);
		
		setJMenuBar(new BaseMenuBar(this));
		
        pack();
        
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;

        setLocation(x, y);
        
        setResizable(false);
        setVisible(true);
	}
}

final class PetConditionSaveListener implements ActionListener {
	
	private PetEditConditionsFrame conditionFrame = null;
	
	private PetSessionProfileFrame sessionFrame = null;
	
	private JTable table = null;
	
	public PetConditionSaveListener(PetEditConditionsFrame conditionFrame, PetSessionProfileFrame sessionFrame, JTable table) {
		this.conditionFrame = conditionFrame;
		this.sessionFrame = sessionFrame;
		this.table = table;
	}
	
	public void actionPerformed(ActionEvent e) {
	    Vector<String[]> newConditionInfo = new Vector<String[]>();

		if(table.getCellEditor() != null) {
			table.getCellEditor().stopCellEditing();
		}
		
	    for(int i = 0; i < table.getRowCount(); i++) {
	    	String conditionName = (String)table.getValueAt(i, 0);
	 /*   	String refScanOnset = (String)table.getValueAt(i, 1);
	    	String numRefScans = (String)table.getValueAt(i, 2);
	    	if(conditionName == null || refScanOnset == null || numRefScans == null) {
	    		continue;
	    	}
	  */
	      	if(conditionName == null || conditionName.length() == 0) {
	    		continue;
	    	}
	      	newConditionInfo.add(new String[]{conditionName});
	    }
	    sessionFrame.updateConditions(newConditionInfo);
	    conditionFrame.dispose();
	}
}
