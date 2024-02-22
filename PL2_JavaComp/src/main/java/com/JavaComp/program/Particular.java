package com.JavaComp.program;

import java.io.Serializable;
import java.util.ArrayList;

public class Particular extends Cliente implements Serializable{
    private String dni;
    
    public Particular(String nombre, String correo, String clave, Direccion direccion, String telefono, int numPedidos, String dni) {
        super(nombre, correo, clave, direccion, telefono, numPedidos);
        this.dni=dni;        
    }
    
    public static void CrearParticular(String nombre, String correo, String clave,
            String telefono, String dni, String titularTarj, String codTarj, 
            String fechTarj, String calleDir, String numDir, 
            String cpDir, String ciudadDir){
        int numPedidos =  0;
        Direccion dir = new Direccion(calleDir, numDir, cpDir, ciudadDir);
        Particular cliente = new Particular(nombre, correo, clave, dir, telefono, numPedidos, dni);
        
        TarjetaCredito tarjeta = new TarjetaCredito(titularTarj, codTarj, fechTarj);
        cliente.setTarjeta(tarjeta);
        ArrayList<Cliente> clientes = DataManager.getClientes();
        clientes.add(cliente);
        DataManager.setClientes(clientes);
        DataManager.setClienteActual(cliente);
    }

    public static void ModificarParticular(String nombre, String correo, String clave,
            String telefono, String dni, String titularTarj, String codTarj, 
            String fechTarj, String calleDir, String numDir, 
            String cpDir, String ciudadDir){
        int numPedidos =  DataManager.getClienteActual().getNumPedidos();
        Direccion dir = new Direccion(calleDir, numDir, cpDir, ciudadDir);
        Particular cliente = new Particular(nombre, correo, clave, dir, telefono, numPedidos, dni);
        TarjetaCredito tarjeta = new TarjetaCredito(titularTarj, codTarj, fechTarj);
        cliente.setTarjeta(tarjeta);
        int i = 0;
        while (!(DataManager.getClientes().get(i).equals(DataManager.getClienteActual()))){
            i++;
        }
        ArrayList<Cliente> clientes = DataManager.getClientes(); 
        clientes.set(i,cliente);
        DataManager.setClientes(clientes);
        DataManager.setClienteActual(cliente);
    }
    
    public String getDni() {
        return dni;
    }

    public void setDni(String dni) {
        this.dni = dni;
    }
}
