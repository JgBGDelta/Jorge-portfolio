/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.JavaComp.program;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.time.LocalDate;
import java.util.ArrayList;

/**
 *
 * @author Slend
 */
public class Pedido implements Serializable {
    private ArrayList<Producto> carrito;
    private Cliente cliente;
    private LocalDate fechaPedido;

    /**
     * Crea el objeto pedido
     * @param cliente el cliente que ha pagado el pedido
     * @param carrito el carrito asociado a la compra
     */
    public Pedido(Cliente cliente, ArrayList<Producto> carrito) {
        this.cliente = cliente;
        this.carrito = carrito;
        fechaPedido = LocalDate.now();
    }
    
    /**
     * Clona un pedido
     * @param pedido el pedido que se va a clonar
     */
    public Pedido(Pedido pedido){
        this.cliente = pedido.getCliente();
        this.carrito = pedido.getCarrito();
        this.fechaPedido = pedido.getFechaPedido();
    }
    
    /**
     * Crea un recibo de texto con los datos del pedido
     */
    public void toText(){
        File textDir = new File("pedidos/" + cliente.getCorreo() + "(" + cliente.getNumPedidos() + ").txt");
        try {textDir.createNewFile();}
        catch (IOException e){System.out.println("El archivo de texto no se pudo crear. Error de I/O: " + e);}
        try {
            FileWriter fileWriter = new FileWriter("pedidos/" + cliente.getCorreo() + "(" + cliente.getNumPedidos() + ").txt");
            String recibo = "RECIBO DE COMPRA\n** " + fechaPedido.toString() + " **\n\n** DATOS DEL CLIENTE **\n\n";
            recibo += "Nombre: " + cliente.getNombre() + "\nCorreo: " + cliente.getCorreo() 
                    + "\nDirección: " + cliente.getDireccion().getCalle() + ", "
                    + cliente.getDireccion().getNumero() + "\n\tCódigo Postal: " 
                    + cliente.getDireccion().getCp() + "\n\tCiudad: " + cliente.getDireccion().getCiudad() 
                    + "\nTeléfono: " + cliente.getTelefono() + "\nTarjeta: **** **** **** " 
                    + cliente.getTarjeta().getCodigo().substring(15) + "\n\n** DATOS DE PRODUCTOS **\n\n";
            for (Producto p : carrito){
                recibo += p.getTitulo().toUpperCase() + ":\nCategoría: " + p.getCategoria() + "\nPrecio: " 
                        + Double.toString(p.getPvp()*p.getStock()) + "€ (" + p.getPvp() + "/ud.)" + 
                        "\nCantidad: " + p.getStock() + "\nCategoría: " + p.getCategoria() + "\n\n";
            }
            recibo += "** RECUENTO DE GASTOS **\n";
            for (Producto p : carrito)
                recibo += p.getTitulo() + ": " + Double.toString(p.getPvp()*p.getStock()) + "€\n";
            recibo += "Envío: 5€\n---------------------------\nTotal: ";
            double sum = 0;
            for (Producto p: carrito)
                sum += p.getPvp()*p.getStock()+5;
            recibo += Double.toString(sum) + "€\n\nGRACIAS POR COMPRAR EN JAVACOMP";
            //Escribe el string recibo en el documento de texto
            fileWriter.write(recibo);
            fileWriter.close();
        } 
        catch (IOException e) {System.out.println("No se pudo escribir en el archivo. Error de I/0: " + e);}
        
    }
    
    /**
     * Finaliza la compra y te a la interfaz de cliente para seguir comprando
     * @param parent frame en el que salen centrados los popup
     * @return devuelve si ha habido un error
     */
    public boolean comprar(javax.swing.JFrame parent){
        ArrayList<Producto> productos = DataManager.getProductos();
        boolean error = false;
        if (carrito.isEmpty()) {
            error = true;
            javax.swing.JOptionPane.showMessageDialog(parent,
                    "Debe añadir al menos un producto al carrito para progresar.", "Error", javax.swing.JOptionPane.WARNING_MESSAGE);
        }
        for (Producto c : carrito){
            boolean existe = false;
            for (Producto p : productos){
                if (p.getTitulo().equals(c.getTitulo()) && p.getStock()>=c.getStock()){
                    p.setStock(p.getStock()-c.getStock());
                    existe = true;
                    break;
                }
            }
            if (!existe){
                error = true;
                javax.swing.JOptionPane.showMessageDialog(parent, "El producto \"" + c.getTitulo() 
                        + "\" está fuera de stock o ha dejado de existir.", "Error de Stock", javax.swing.JOptionPane.WARNING_MESSAGE);
            }
        }
        if (!error){
            DataManager.getPedidos().add(this);
            cliente.setNumPedidos(cliente.getNumPedidos()+1);
            javax.swing.ImageIcon icon = new javax.swing.ImageIcon("src/main/resources/images/LogoJavaComp.png");
            javax.swing.JOptionPane.showMessageDialog(parent, "Compra realizada con éxito. \n Gracias por comprar en JavaComp",
                    "Compra realizada con éxito", javax.swing.JOptionPane.INFORMATION_MESSAGE, icon);
        }
        return error;
    }
    
    public ArrayList<Producto> getCarrito() {
        return carrito;
    }

    public void setCarrito(ArrayList<Producto> carrito) {
        this.carrito = carrito;
    }

    public Cliente getCliente() {
        return cliente;
    }

    public void setCliente(Cliente cliente) {
        this.cliente = cliente;
    }

    public LocalDate getFechaPedido() {
        return fechaPedido;
    }

    public void setFechaPedido(LocalDate fechaPedido) {
        this.fechaPedido = fechaPedido;
    }
    
    
}
