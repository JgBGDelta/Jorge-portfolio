package com.JavaComp.program;

import java.io.Serializable;
import java.time.LocalDate;


public class TarjetaCredito implements Serializable{
    private String titular;
    private String codigo;
    private String caducidad;

    public TarjetaCredito(String titular, String codigo, String caducidad) {
        this.titular = titular;
        this.codigo = codigo;
        this.caducidad = caducidad;
    }

    public String getTitular() {
        return titular;
    }

    public void setTitular(String titular) {
        this.titular = titular;
    }

    public String getCodigo() {
        return codigo;
    }

    public void setCodigo(String codigo) {
        this.codigo = codigo;
    }

    public String getCaducidad() {
        return caducidad;
    }

    public void setCaducidad(String caducidad) {
        this.caducidad = caducidad;
    }
    
    /**
     * Verifica si una fecha de tipo mm/aaaa es correcta y vigente
     * @param fecha el string con la fecha en formato mm/aaaa
     * @return si la tarjeta es vigente en el array [0] devuelve true y si el mes es correcto en el array [1] devuelve true
     */
    public static boolean[] comprobarFechaTarjeta (String fecha) throws java.lang.NumberFormatException{
             boolean tarjetaVigente=false;
             boolean mesValido=false;
             String [] partes = fecha.split("/");
             int mes = Integer.parseInt(partes[0]);
             int anno = Integer.parseInt(partes[1]);
             LocalDate fechaActual = LocalDate.now();
             if (mes >= 1 && mes <= 12){
                mesValido=true;
             }
    
             if (anno>fechaActual.getYear()||(anno==fechaActual.getYear() && mes>fechaActual.getMonthValue())){                 
            tarjetaVigente=true;
             }
             boolean[] booleanArray = new boolean[2];
             booleanArray[0] = tarjetaVigente;
             booleanArray[1] = mesValido;
             return (booleanArray);
    }
}
