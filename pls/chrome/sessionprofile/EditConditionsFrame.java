package pls.chrome.sessionprofile;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.border.Border;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellRenderer;
import javax.swing.table.TableColumn;

import nl.jj.swingx.gui.modal.JModalFrame;
import pls.chrome.shared.BaseMenuBar;

@SuppressWarnings("serial")
public class EditConditionsFrame extends JModalFrame {
	
	private boolean isNPAIRSAnalysis;
	
	public EditConditionsFrame(SessionProfileFrame sessionFrame, boolean isNPAIRSAnalysis) {
		super("Edit Conditions");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
		addAdditionalModalToWindow(sessionFrame);
		
		this.isNPAIRSAnalysis = isNPAIRSAnalysis;
	    
	    DefaultTableModel model = new DefaultTableModel();
	    JTable table = createTable();
	    JScrollPane sp;
	    
	    table.setModel(model);
	    table.setPreferredScrollableViewportSize(
	    		new Dimension(table.getPreferredScrollableViewportSize().width,
	    				160));

	    loadTableModel(sessionFrame, isNPAIRSAnalysis, model);
	    table.getModel().addTableModelListener(new ModelListener(table));
	    table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
	    sp = new JScrollPane(table);
	    resizeTable(table);
	    
        JButton saveButton = new JButton("Store And Continue", 
        		new ImageIcon(this.getClass().getResource("/toolbarButtonGraphics/general/SendMail16.gif")));
        saveButton.setIconTextGap(15);
        saveButton.addActionListener(new ConditionSaveListener(this, sessionFrame, table));
        
		setLayout(new BorderLayout());
		add(sp, BorderLayout.NORTH);
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
	
	/**
	 * Create the table for the edit conditions frame.
	 * @return the created table.
	 */
	private JTable createTable(){
		
		JTable table = new JTable(){
			SelectedRenderer sr = new SelectedRenderer();
			public TableCellRenderer getCellRenderer(int row, int column) {
				int focusR = this.getSelectionModel().getLeadSelectionIndex();
			    int focusC = this.getColumnModel().getSelectionModel()
			    									.getLeadSelectionIndex();
				
			    //Return a special renderer for the selected cell.
		        if ((row == focusR) && (column == focusC)) {
		            return sr;
		        }
		        
		        return super.getCellRenderer(row, column);
		    }
		};
		
		return table;
	}
	/**
	 * Load the table with data.
	 * @param sessionFrame the session frame stores the condition info to load.
	 * @param isNPAIRSAnalysis are we adding extra columns 
	 * if this isn't an npairs analysis?
	 * @param model the table model to load data for.
	 */
	private void loadTableModel(SessionProfileFrame sessionFrame,
			boolean isNPAIRSAnalysis, DefaultTableModel model) {
		model.addColumn("Condition Name");
	    if (!isNPAIRSAnalysis) {
	    	model.addColumn("Reference Scan Onset");
	    	model.addColumn("Number of Reference Scans");
	    }
	    
	    int numRows = 0;
	    for(String[] conditionRow : sessionFrame.conditionInfo) {
	    	model.addRow(conditionRow);
	    	numRows++;
	    }
	    
	    int numColumns = model.getColumnCount();
	    Object[] blankRow = new Object[numColumns];
	    
	    for(int i = 0; i < numColumns; i++){
	    	blankRow[i] = null;
	    }
	    
	    for(; numRows < 100; numRows++) {
	    	model.addRow(blankRow);
	    }
	}

	/**
	 * When the user adds a new value to the table, resize
	 * the column so that the name is not truncated.
	 * @param table the table to resize.
	 */
	private void resizeTable(JTable table){
		int cols = table.getColumnCount();
		int rows = table.getRowCount();
		int maxWidth = 0;

		/*
		 * Go through each column and calculate the maximum possible width.
		 * When this is found set the column to that width.
		 */
		for(int c = 0; c < cols; c++){
			maxWidth = 0;
			for(int r = 0; r < rows; r++){
				Object val = table.getValueAt(r,c);
				TableCellRenderer rend = table.getCellRenderer(r,c);
				Component cell = rend.getTableCellRendererComponent(table, 
													val, false,false, r,c);
				maxWidth = (maxWidth > cell.getPreferredSize().width) ? 
							maxWidth : cell.getPreferredSize().width;
				
			}
			//Investigate the width of the column header.
			
			TableColumn header;
			TableCellRenderer headerRend;
			Component headerCell;
			header = table.getTableHeader().getColumnModel().getColumn(c);
			headerRend = header.getHeaderRenderer();
			
			if(headerRend == null) 
				headerRend = table.getTableHeader().getDefaultRenderer();
			
			headerCell = headerRend.getTableCellRendererComponent(table,
								header.getHeaderValue(), false, false, 0, c);
			
			maxWidth = (maxWidth > headerCell.getPreferredSize().width) ?
						maxWidth : headerCell.getPreferredSize().width;
			
			/*We need the 2 extra pixels otherwise the characters are still
			 *truncated.  
			 */
			header.setPreferredWidth(maxWidth+2);
		}
	}
	/**
	 * A customized renderer for when a cell is selected in the conditions
	 * table. This renderer is needed so the user can actually see which
	 * cell is highlighted. On this machine, with the default colour scheme 
	 * the border that is drawn is shadowed by the background colour of the
	 * selected item. In simpler terms this class is needed so we can see 
	 * which cell has focus.
	 */
	private class SelectedRenderer extends JLabel implements TableCellRenderer{
		Border highlight;
		
		SelectedRenderer(){
			setOpaque(true); //needed so the background is visible.
		}
		
		public Component getTableCellRendererComponent(JTable table,
				Object val, boolean isSelected, 
				boolean hasFocus, int row, int col) {
			

			if(highlight == null){
				highlight = BorderFactory.createMatteBorder(1,1,1,1,
						Color.black);
			}
			setBorder(highlight);
			setBackground(table.getSelectionBackground());
			
			setText((String)val);
			
			return this;
		}
	}
	
	private class ModelListener implements TableModelListener{
		JTable tableref;
		
		ModelListener(JTable table){
			tableref = table;
		}
		
		@Override
		public void tableChanged(TableModelEvent e) {
			int row = e.getFirstRow();
			int col = e.getColumn();
			Object modifedVal = tableref.getValueAt(row, col);
			
			if(isNPAIRSAnalysis){
				resizeTable(tableref);
				return;
			}
			
			/*
			 * Check that a condition name was entered (column 0) and that the
			 * user actually entered something (null if didnt enter anything,
			 * "" if they entered a blank string). Set default valuse for the
			 * new condition.
			 */
			if(col == 0 && modifedVal != null 
				&& !((String) modifedVal).equals("")){
				if (tableref.getValueAt(row,col+1) == null &&
					tableref.getValueAt(row, col+2) == null){
					tableref.setValueAt("0", row,col+1);
					tableref.setValueAt("1", row,col+2);
				}
			}
			resizeTable(tableref);
		}
	}

final class ConditionSaveListener implements ActionListener {
	
	private EditConditionsFrame conditionFrame = null;
	
	private SessionProfileFrame sessionFrame = null;
	
//	private NpairsSessionProfileFrame nSessionFrame = null;
	
	private JTable table = null;
	
	public ConditionSaveListener(EditConditionsFrame conditionFrame, SessionProfileFrame sessionFrame, JTable table) {
		this.conditionFrame = conditionFrame;
		this.sessionFrame = sessionFrame;
		this.table = table;
	}
	
//	public ConditionSaveListener(EditConditionsFrame conditionFrame, NpairsSessionProfileFrame nSessionFrame, JTable table) {
//		this.conditionFrame = conditionFrame;
//		this.nSessionFrame = nSessionFrame;
//		this.table = table;
//	}
//	
	public void actionPerformed(ActionEvent e) {
	    Vector<String[]> newConditionInfo = new Vector<String[]>();

		if(table.getCellEditor() != null) {
			table.getCellEditor().stopCellEditing();
		}
		
		if (!conditionFrame.isNPAIRSAnalysis) {
			boolean defaultRefInfo = true;
			for(int i = 0; i < table.getRowCount(); i++) {
		    	String conditionName = (String)table.getValueAt(i, 0);
		    	String refScanOnset = (String)table.getValueAt(i, 1);
		    	String numRefScans = (String)table.getValueAt(i, 2);
		    	if (conditionName == null || refScanOnset == null || numRefScans == null
		    		|| conditionName.length() == 0 || refScanOnset.length() == 0 || numRefScans.length() == 0) {
		    		continue;
		    	}
		    	if (!refScanOnset.equals("0") || !numRefScans.equals("1")) {
		    		defaultRefInfo = false;
		    		refScanOnset = "0";
		    		numRefScans = "1";
		    	}
		    	newConditionInfo.add(new String[]{conditionName, refScanOnset, numRefScans});
		    }
			if (!defaultRefInfo) {
				// only default ref scan info (0 1) has been tested so complain if anything 
	    		// else is entered
	    		// TODO: re-enable custom ref info when it works
		    
		    String refSettingWarnMess = "Condition Info: Custom reference scan settings cannot " +
		        "currently be used.\nSetting reference scan onset and block size to default " +
		        "values 0 and 1, respectively, instead.";
		    JOptionPane.showMessageDialog(null, refSettingWarnMess, "Warning", 
		    		JOptionPane.WARNING_MESSAGE);
			}
		} else {
			for(int i = 0; i < table.getRowCount(); i++) {
		    	String conditionName = (String)table.getValueAt(i, 0);
		    	if (conditionName == null || conditionName.length() == 0) {
			    	continue;
			    }
		    	// set ref scan onset/length to 0 and 1 respectively by default
			    newConditionInfo.add(new String[]{conditionName, "0", "1"});
		    }
		}
		
//	    if (sessionFrame != null) {
	    	sessionFrame.updateConditions(newConditionInfo);
//	    }
//	    else {
//	    	nSessionFrame.updateConditions(newConditionInfo);
//	    }
	    conditionFrame.dispose();
	}
}
}
