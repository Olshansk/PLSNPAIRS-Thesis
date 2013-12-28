package pls.chrome.analysis;

import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.BorderLayout;
import java.awt.Toolkit;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.JButton;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.BorderFactory;
import javax.swing.JCheckBox;

import nl.jj.swingx.gui.modal.JModalFrame;

import pls.chrome.shared.BaseMenuBar;
import pls.shared.MLFuncs;

import com.jmatio.types.MLCell;
import com.jmatio.types.MLStructure;

import extern.NewMatFileReader;


public class DeselectConditionsFrame  extends JModalFrame {
	
	protected Vector conditionSelection = null;
	
	public DeselectConditionsFrame(JFrame parent, Vector<String[]> sessionProfiles, 
			Vector<Integer> conditionSelection) {
		
		super("Deselect Conditions");
	    setDefaultCloseOperation(DISPOSE_ON_CLOSE);
	    addAdditionalModalToWindow(parent);
	    
	    this.conditionSelection = conditionSelection;
	    
		if(sessionProfiles.size() == 0) {
			JOptionPane.showMessageDialog(null, "Group needs to be added before you can " +
					"deselect conditions.", "Error", JOptionPane.ERROR_MESSAGE);
			return;
		}
	    
	    MLStructure sessionInfo = null;
	    try {
	    	
	    	sessionInfo = (MLStructure)new NewMatFileReader(
	    			sessionProfiles.get(0)[0]).getContent().get("session_info");
	    	
	    } catch(Exception ex) {
	    	
	    	JOptionPane.showMessageDialog(null, 
	    			"Session information could not be read from file " 
	    			+ sessionProfiles.get(0)[0] + ".", 
	    			"Error", JOptionPane.ERROR_MESSAGE);
	    	return;
	    	
	    }
	    String[] conditions = MLFuncs.MLCell1dRow2StrArray((MLCell)sessionInfo.getField("condition"));
	    	    
	    setJMenuBar(new BaseMenuBar(this));

	    add(new ConditionSelectionPanel(this, conditions, conditionSelection));
	    
        // Display the window
        pack();
	    setResizable(false);
        
        // Position the frame on the screen
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (int)(screen.getWidth() - getWidth()) / 2;
        int y = (int)(screen.getHeight() - getHeight()) / 2;
        setLocation(x, y);
        
        this.pack();
        
        setVisible(true);
	}
}

final class ConditionSelectionPanel extends JPanel {
	
	public ConditionSelectionPanel(JFrame frame, String[] conditions, Vector<Integer> conditionSelection) {
		final JCheckBox[] options = new JCheckBox[conditions.length];
		if(conditionSelection.isEmpty()) {
			for(String c : conditions) {
				conditionSelection.add(new Integer(1));
			}
		}
		
		JPanel boxPanel = new JPanel(new GridLayout(5, 0));
		for(int i = 0; i < conditions.length; i++) {
			options[i] = new JCheckBox(conditions[i]);
			if(conditionSelection.get(i).intValue() == 1) {
				options[i].setSelected(true);
			}
			boxPanel.add(options[i]);
		}
		JScrollPane boxScrollPane = new JScrollPane(boxPanel);
		//boxScrollPane.setPreferredSize(new Dimension(550, 50));

		
		JPanel buttonPanel = new JPanel(new GridLayout(1, 0));
				
		JButton clearAll = new JButton("Clear All");
		clearAll.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
					for(JCheckBox option : options) {
						option.setSelected(false);
					}
				}
			});

		JButton selectAll = new JButton("Select All");
		selectAll.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
					for(JCheckBox option : options) {
						option.setSelected(true);
					}
				}
			});
		
		JButton saveSelection = new JButton("Save Selection");
		saveSelection.addActionListener(new SaveConditionsListener(frame, conditionSelection, options));
		
		buttonPanel.add(clearAll);
		buttonPanel.add(selectAll);
		buttonPanel.add(saveSelection);
		buttonPanel.setBorder(BorderFactory.createEmptyBorder(10, 0, 0, 0));
		
		JPanel mainPanel = new JPanel(new BorderLayout());

		mainPanel.add(boxScrollPane, BorderLayout.NORTH);
		mainPanel.add(buttonPanel, BorderLayout.SOUTH);
	    //setPreferredSize(new Dimension(600, 100));
		add(mainPanel);
		
	}
}

final class SaveConditionsListener implements ActionListener {
	
	private JFrame frame = null;
	
	private Vector<Integer> conditionSelection = null;
	
	private JCheckBox[] options = null;
	
	public SaveConditionsListener(JFrame frame, Vector<Integer> conditionSelection, JCheckBox[] options) {
		this.frame = frame;
		this.conditionSelection = conditionSelection;
		this.options = options;
	}
	public void actionPerformed(ActionEvent e) {
		conditionSelection.removeAllElements();
		for(int i = 0; i < options.length; i++) {
			if(options[i].isSelected()) {
				conditionSelection.add(new Integer(1));
			} else {
				conditionSelection.add(new Integer(0));
			}
		}
		frame.dispose();
		
	}
}