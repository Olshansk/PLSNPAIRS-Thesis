package pls.chrome.result.view;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JRadioButton;

import pls.chrome.result.model.GeneralRepository;

public class RegressionOptions extends JDialog {
	GeneralRepository repo = null;
	private boolean meanOrMedian = true; //mean = t; median = f
	private double[] means;
	private double[] medians;
	JRadioButton rMean = null;
	JRadioButton rMedian = null;
	JDialog predReg = null;
	private RegressionOptions me = this;
	
	public RegressionOptions(GeneralRepository repository) {
		repo = repository;
	}
	
	public void initialize(double[] means, double[] medians) {
		JLabel mOrM = new JLabel("Regress against: ");
		ButtonGroup group = new ButtonGroup();
		rMean = new JRadioButton("Means");
		rMean.setSelected(true);
		rMedian = new JRadioButton("Medians");
		group.add(rMean);
		group.add(rMedian);
		rMean.addActionListener(new RadioListener());
		rMedian.addActionListener(new RadioListener());
		
		JPanel cf = (JPanel) getContentPane();
		cf.setLayout(new BorderLayout());
		
		JPanel top = new JPanel();
		
		top.add(mOrM);
		top.add(rMean);
		top.add(rMedian);
		
		cf.add(top, BorderLayout.NORTH);
		
		JButton loadData = new JButton("Load .txt file for regression");
		loadData.addActionListener(new LoadListener());
		
		cf.add(loadData, BorderLayout.CENTER);
		cf.add(new JLabel("Note: The text file must contain only one column, and the same number of elements as this result has subjects."), BorderLayout.SOUTH);
		
		pack();
		setVisible(true);
		
		this.means = means;
		this.medians = medians;
		
	}
	
	private class RadioListener implements ActionListener {

		@Override
		public void actionPerformed(ActionEvent e) {
			if (e.getSource().equals(rMedian)) {
				meanOrMedian = false;
			} else if (e.getSource().equals(rMean)) {
				meanOrMedian = true;
			}			
		}
		
	}
	
	private class LoadListener implements ActionListener {
		@Override
		public void actionPerformed(ActionEvent arg0) {
			final JFileChooser fc = new JFileChooser(".");
			fc.showOpenDialog(me);
			File f = fc.getSelectedFile();
			
			try {
				if (meanOrMedian) {
					//PredictionRegression pr = new PredictionRegression(repo, means, meanOrMedian, new double[]{3, 45, 1, 23, 33, 66, 77, 11});
					PredictionRegression pr = new PredictionRegression(repo, means, meanOrMedian, getData(f));
					
				} else {
					//PredictionRegression pr = new PredictionRegression(repo, medians, meanOrMedian, new double[]{3, 45, 1, 23, 33, 66, 77, 11});
					PredictionRegression pr = new PredictionRegression(repo, medians, meanOrMedian, getData(f));
				}
			} catch (NumberFormatException e) {
				System.out.println("There was a problem with loading data from the file you just chose.");
				e.printStackTrace();
			} catch (IllegalArgumentException e) {
				System.out.println("There was a problem loading the file you just chose.");
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				System.out.println("There was a problem loading the file you just chose.");
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			dispose();
		}
		
		public double[] getData(File f) throws NumberFormatException, IOException {
			
			if (f != null) {
				ArrayList<Double> dataL = new ArrayList<Double>();
				BufferedReader reader = new BufferedReader(new FileReader(f));
				String line = reader.readLine();
				while (line != null) {
					dataL.add(new Double(line));
					line = reader.readLine();
				}
				Double[] data1 = new Double[0];
				data1 = dataL.toArray(data1);
				double[] data2 = new double[data1.length];
				for (int i = 0; i < data1.length; i++) {
					data2[i] = data1[i].doubleValue();
				}
				
				return data2;
			}
			return null;
			
		}
	}

}
