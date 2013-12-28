/*Copyright (c) 2006, Wojciech Gradkowski
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright 
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the 
      distribution.
    * Neither the name of the JMatIO nor the names of its 
      contributors may be used to endorse or promote products derived 
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

package extern;

/**
 * This class is a modified version of the original class for 
 * MAT-file data types (com.jmatio.common.MatDataTypes).
 * It now includes additional information on singles.
 * 
 * @author Wojciech Gradkowski <wgradkowski@gmail.com>
 * @version 1.1 Modifications by Felix Wong
 */
public class NewMatDataTypes
{
    /* MAT-File Data Types */
    public static final int miUNKNOWN   = 0;
    public static final int miINT8      = 1;
    public static final int miUINT8     = 2;
    public static final int miINT16     = 3;
    public static final int miUINT16    = 4;
    public static final int miINT32     = 5;
    public static final int miUINT32    = 6;
    public static final int miSINGLE    = 7;
    public static final int miDOUBLE    = 9;
    public static final int miINT64     = 12;
    public static final int miUINT64    = 13;
    public static final int miMATRIX    = 14;
    public static final int miCOMPRESSED    = 15;
    public static final int miUTF8      = 16;
    public static final int miUTF16     = 17;
    public static final int miUTF32     = 18;

    public static final int miSIZE_INT32    = 4;
    public static final int miSIZE_INT16    = 2;
    public static final int miSIZE_INT8     = 1;
    public static final int miSIZE_UINT32   = 4;
    public static final int miSIZE_UINT16   = 2;
    public static final int miSIZE_UINT8    = 1;
    public static final int miSIZE_SINGLE   = 4;
    public static final int miSIZE_DOUBLE   = 8;
    public static final int miSIZE_CHAR     = 1;
    
    /**
     * Return number of bytes for given type.
     * 
     * @param type - <code>NewMatDataTypes</code>
     * @return
     */
    public static int sizeOf(int type)
    {
        switch ( type )
        {
            case NewMatDataTypes.miINT8:
                return miSIZE_INT8;
            case NewMatDataTypes.miUINT8:
                return miSIZE_UINT8;
            case NewMatDataTypes.miINT16:
                return miSIZE_INT16;
            case NewMatDataTypes.miUINT16:
                return miSIZE_UINT16;
            case NewMatDataTypes.miINT32:
                return miSIZE_INT32;
            case NewMatDataTypes.miUINT32:
                return miSIZE_UINT32;
            case NewMatDataTypes.miSINGLE:
                return miSIZE_SINGLE;
            case NewMatDataTypes.miDOUBLE:
                return miSIZE_DOUBLE;
            default:
                return 1;
        }
    }
    /**
     * Get String representation of a data type
     * 
     * @param type - data type
     * @return - String representation
     */
    public static String typeToString(int type)
    {
        String s;
        switch (type)
        {
            case NewMatDataTypes.miUNKNOWN:
                s = "unknown";
                break;
            case NewMatDataTypes.miINT8:
                s = "int8";
                break;
            case NewMatDataTypes.miUINT8:
                s = "uint8";
                break;
            case NewMatDataTypes.miINT16:
                s = "int16";
                break;
            case NewMatDataTypes.miUINT16:
                s = "uint16";
                break;
            case NewMatDataTypes.miINT32:
                s = "int32";
                break;
            case NewMatDataTypes.miUINT32:
                s = "uint32";
                break;
            case NewMatDataTypes.miSINGLE:
                s = "single";
                break;
            case NewMatDataTypes.miDOUBLE:
                s = "double";
                break;
            case NewMatDataTypes.miINT64:
                s = "int64";
                break;
            case NewMatDataTypes.miUINT64:
                s = "uint64";
                break;
            case NewMatDataTypes.miMATRIX:
                s = "matrix";
                break;
            case NewMatDataTypes.miCOMPRESSED:
                s = "compressed";
                break;
            case NewMatDataTypes.miUTF8:
                s = "uft8";
                break;
            case NewMatDataTypes.miUTF16:
                s = "utf16";
                break;
            case NewMatDataTypes.miUTF32:
                s = "utf32";
                break;
            default:
                s = "unknown";
        }
        return s;
    }
    
}