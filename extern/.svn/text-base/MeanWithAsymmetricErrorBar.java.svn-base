/* ===========================================================
 * JFreeChart : a free chart library for the Java(tm) platform
 * ===========================================================
 *
 * (C) Copyright 2000-2008, by Object Refinery Limited and Contributors.
 *
 * Project Info:  http://www.jfree.org/jfreechart/index.html
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
 * USA.
 *
 * [Java is a trademark or registered trademark of Sun Microsystems, Inc.
 * in the United States and other countries.]
 *
 * -----------------------------
 * MeanAndStandardDeviation.java
 * -----------------------------
 * (C) Copyright 2003-2008, by Object Refinery Limited.
 *
 * Original Author:  David Gilbert (for Object Refinery Limited);
 * Contributor(s):   -;
 *
 * Changes:
 * --------
 * 05-Feb-2002 : Version 1 (DG);
 * 05-Feb-2005 : Added equals() method and implemented Serializable (DG);
 * 02-Oct-2007 : Added getMeanValue() and getStandardDeviationValue() methods
 *               for convenience, and toString() method for debugging (DG);
 * 02-Feb-2010 : Modified into the class you now see here.
 *
 */

package extern;

import java.io.Serializable;

import org.jfree.data.statistics.MeanAndStandardDeviation;
import org.jfree.util.ObjectUtilities;

public class MeanWithAsymmetricErrorBar implements Serializable {

    /** For serialization. */
    private static final long serialVersionUID = 7413468697315721515L;
	
    /** The mean. */
    private Number mean;

    /** The upper value. */
    private Number upperValue;
    
    /** The upper value. */
    private Number lowerValue;
    
    public MeanWithAsymmetricErrorBar(double mean, double upperValue, double lowerValue) {
        this(new Double(mean), new Double(upperValue), new Double(lowerValue));
    }
    
    /**
     * Creates a new mean, upper and lower bound record.
     */
    public MeanWithAsymmetricErrorBar(Number mean, Number upperValue, Number lowerValue) {
        this.mean = mean;
        this.upperValue = upperValue;
        this.lowerValue = lowerValue;
    }
    
    /**
     * Returns the mean.
     *
     * @return The mean.
     */
    public Number getMean() {
        return this.mean;
    }
    
    /**
     * Returns the mean as a double primitive.  If the underlying mean is
     * <code>null</code>, this method will return <code>Double.NaN</code>.
     *
     * @return The mean.
     *
     * @see #getMean()
     *
     * @since 1.0.7
     */
    public double getMeanValue() {
        double result = Double.NaN;
        if (this.mean != null) {
            result = this.mean.doubleValue();
        }
        return result;
    }
    
    /**
     * Returns the upper value.
     *
     * @return The upper value of CI.
     */
    public Number getUpper() {
        return this.upperValue;
    }
    
    /**
     * Returns the upper value as a double primitive.  If the underlying
     * upper value is <code>null</code>, this method will return
     * <code>Double.NaN</code>.
     *
     * @return The upper value of CI.
     */
    public double getUpperValue() {
        double result = Double.NaN;
        if (this.upperValue != null) {
            result = this.upperValue.doubleValue();
        }
        return result;
    }
	
    /**
     * Returns the lower value.
     *
     * @return The lower value of CI.
     */
    public Number getLower() {
        return this.lowerValue;
    }
    
    /**
     * Returns the lower value as a double primitive.  If the underlying
     * lower value is <code>null</code>, this method will return
     * <code>Double.NaN</code>.
     *
     * @return The lower value of CI.
     */
    public double getLowerValue() {
        double result = Double.NaN;
        if (this.lowerValue != null) {
            result = this.lowerValue.doubleValue();
        }
        return result;
    }
    /**
     * Tests this instance for equality with an arbitrary object.
     * @param obj  the object (<code>null</code> permitted).
     * @return A boolean.
     */
    public boolean equals(Object obj) {
        if (obj == this) {
            return true;
        }
        if (!(obj instanceof MeanWithAsymmetricErrorBar)) {
            return false;
        }
        MeanWithAsymmetricErrorBar that = (MeanWithAsymmetricErrorBar) obj;
        if (!ObjectUtilities.equal(this.mean, that.mean)) {
            return false;
        }
        if (!ObjectUtilities.equal(
            this.upperValue, that.upperValue)
        ) {
            return false;
        }
        if (!ObjectUtilities.equal(
                this.lowerValue, that.lowerValue)
            ) {
                return false;
        }
        return true;
    }

    /**
     * Returns a string representing this instance.
     *
     * @return A string.
     *
     * @since 1.0.7
     */
    public String toString() {
        return "[" + this.mean + ", " + this.upperValue + ", " + this.lowerValue + "]";
    }
	
}

	

        
   
	 
   
      

   
    
    


