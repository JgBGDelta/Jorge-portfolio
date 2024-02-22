/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package com.JavaComp.program;

import com.JavaComp.interf.DisplayInventario;
import com.JavaComp.interf.DisplayProdCarro;
import com.JavaComp.interf.DisplayProducto;
import com.JavaComp.interf.DisplayVenta;
import com.JavaComp.interf.InterfCarro;
import com.JavaComp.interf.Inventario;
import static com.JavaComp.program.Producto.crearPanel;
import java.io.*;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Comparator;
import javax.swing.JPanel;

public class DataManager {
    
    private static Cliente clienteActual = null;
    private static ArrayList<Cliente> clientes = new ArrayList();
    private static ArrayList<Producto> productos = new ArrayList();
    private static ArrayList<Producto> carritoActual = new ArrayList();
    private static ArrayList<Pedido> pedidos = new ArrayList();

    public static Cliente getClienteActual() {
        return clienteActual;
    }
    
    /**
     * filtra el ArrayList productos por la clave de busqueda busqueda y devuelve un ArrayList
     * con todos los productos que tengan esos terminos.
     * @param busqueda
     * @return El ArrayList productos filtrado por la clave busqueda
     */
    public static ArrayList busquedaProducto(String busqueda){
        ArrayList<Producto> filtrado = new ArrayList();
        String[] arr = busqueda.split(" ");
        for (String i: arr){
            for (Producto prod: productos){
                if (!filtrado.contains(prod)){
                    boolean encontrado = false;
                    if (prod.getCaracteristicas().toLowerCase().contains(i.toLowerCase())
                            || prod.getTitulo().toLowerCase().contains(i.toLowerCase())
                            || prod.getCategoria().toLowerCase().contains(i.toLowerCase())){
                        filtrado.add(prod);
                        encontrado = true;
                    }
                    else{
                        for(Opinion op: prod.getOpiniones()){
                            if(op.getComentario().toLowerCase().contains(i.toLowerCase()) && !encontrado){
                                filtrado.add(prod);
                                encontrado = true;
                            }
                        }
                    }
                }
            }
        }
        
        return filtrado;
    }
    
    /**
     * Ordena un ArrayList de productos por precio
     * @param reverse if true ordena a la inversa
     * @param array el arraylist a ordenar
     */
    public static void sortPrecio(boolean reverse, ArrayList<Producto> array){
        if(!reverse){
            array.sort(new Comparator<Producto>() {
                @Override
                public int compare(Producto p1, Producto p2) {
                    return Double.compare(p1.getPvp(), p2.getPvp());
                }
            }.reversed());
        }
        else {
            array.sort(new Comparator<Producto>() {
                @Override
                public int compare(Producto p1, Producto p2) {
                    return Double.compare(p1.getPvp(), p2.getPvp());
                }
            });
        }
        
    }
    
    /**
     * Ordena un ArrayList de productos por titulo
     * @param array el arraylist a ordenar
     */
    public static void sortTitulo(ArrayList<Producto> array){
        array.sort(new Comparator<Producto>(){
            @Override
            public int compare(Producto p1, Producto p2){
                return p1.getTitulo().compareTo(p2.getTitulo());
            }
        });
    }
    
    /**
     * Ordena el array por relevancia (por mayor puntuacion de opinion media)
     * @param array el array que se va a ordenar
     */
    public static void sortRelevancia(ArrayList<Producto> array) {
        array.sort(new Comparator<Producto>(){
            @Override
            public int compare(Producto p1, Producto p2){
                return Float.compare(p1.getMidRating(), p2.getMidRating());
            }
        }.reversed());
    }
    
    /**
     * Mete un producto al carrito
     * @param prod el producto que se va a añadir
     */
    public static void meterAlCarro(Producto prod){
        carritoActual.add(prod);
    }
    
    /**
     * Quita un producto del carrito
     * @param prod el producto que se va a quitar
     */
    public static void quitarDeCarro(Producto prod){
        int i = 0;
        while(!carritoActual.get(i).equals(prod)) i++;
        carritoActual.remove(i);
    }
    
    /**
     * Filtra un ArrayList de productos por categoría
     * @param categoria la categoría por la que se va a filtrar
     * @param array el array que se va a filtrar
     * @return El arraylist ya filtrado
     */
    public static ArrayList filtrarCategoria(String categoria, ArrayList<Producto> array){
        ArrayList listaFiltrada = new ArrayList();
        if(!categoria.equals("Todos")){
            for(Producto prod : array){
                if (prod.getCategoria().equals(categoria)){
                    listaFiltrada.add(prod);
                }
            }
            return listaFiltrada;
        }
        else return array;
    }
    
    /**
     * Filtra un array de pedidos, dejando los que se han a partir de una fecha
     * @param isAfter true si filtras fechas posteriores a la fecha estipulada
     * @param fecha la fecha por la que se va a filtrar
     * @param array el array de pedidos que se va a filtrar
     * @return El array filtrado por fecha
     */
    public static ArrayList filtrarFecha(boolean isAfter, LocalDate fecha, ArrayList<Pedido> array){
        ArrayList<Pedido> newArray = new ArrayList();
        if (isAfter)
            for (Pedido ped : array){
                if (ped.getFechaPedido().isAfter(fecha)){
                    newArray.add(ped);
                }
            }
        else {
            for (Pedido ped : array){
                if (ped.getFechaPedido().isBefore(fecha)){
                    newArray.add(ped);
                }
            }
        }
        return newArray;
    }
    
    /**
     * Crea un DisplayProducto por cada producto en el array y los muestra en el panel
     * @param lista el array con los productos
     * @param panel el panel en el que se muestran los productos
     */
    public static void displayList(ArrayList<Producto> lista, JPanel panel){
        panel.removeAll();
        for (Producto prod : lista){
            if (prod.getStock() > 0){
                DisplayProducto display = crearPanel(prod);
                display.setVisible(true);
                panel.add(display);
            }
        }
        panel.repaint();
        panel.revalidate();
    }
    
    /**
     *Crea un DisplayVenta por cada pedido en el array y los muestra en el panel
     * @param lista la lista de pedidos
     * @param panel el panel en el que se van a mostrar los displays
     */
    public static void displayVenta(ArrayList<Pedido> lista, JPanel panel){
        panel.removeAll();
        for (Pedido ped : lista){
            DisplayVenta display = new DisplayVenta(ped);
            display.setVisible(true);
            panel.add(display);
            
        }
        panel.repaint();
        panel.revalidate();
    }
    
    /**
     * Crea un DisplayInventario por cada producto en la ArrayList lista y los mete en el JPanel panel
     * Adicionalmente vacia panel y lo repinta
     * @param lista el ArrayList con la lista de productos que se van a mostrar
     * @param panel el panel que va a contener a los displays
     * @param parent el frame superior que contiene todos los componentes
     */
    public static void displayInventario(ArrayList<Producto> lista, JPanel panel , Inventario parent){
        panel.removeAll();
        for (int i = 0; i < lista.size(); i++){
            DisplayInventario display = Producto.crearPanelInventario(lista.get(i), parent);
            display.setVisible(true);
            panel.add(display);
            
        }
        panel.repaint();
        panel.revalidate();
    }
    
    /**
     * Crea un DisplayProdCarro por cada producto existente en carritoActual
     * Adicionalmente asigna el parámetro parent del DisplayProdCarro a el IntefCarro que lo contiene
     * @param panel
     * @param interfaz
     */
    public static void displayCarrito(JPanel panel, InterfCarro interfaz){
        panel.removeAll();
        for (int i = 0; i < carritoActual.size(); i++){
            DisplayProdCarro display = Producto.crearPanelCarro(carritoActual.get(i));
            display.setParent(interfaz);
            display.setVisible(true);
            panel.add(display);
            
        }
        panel.repaint();
        panel.revalidate();
    }
    
    public static void setClienteActual(Cliente clienteActual) {
        DataManager.clienteActual = clienteActual;
    }
    
    /**
     * Guarda los datos del ArrayList i en un archivo con la dirección saveDir
     * El archivo se guarda en un archivo binario
     * @param i
     * @param saveDir
     */
    public static void uploadSave(ArrayList i, String saveDir){
        try{
            FileOutputStream oStream = new FileOutputStream(saveDir);
            ObjectOutputStream ooStream = new ObjectOutputStream(oStream);
            ooStream.writeObject(i);
            oStream.close();
        }
        catch (IOException ioe){
            System.out.println("Error de I/O: " + ioe.getMessage());
        }
        catch (Exception e){
            System.out.println("Error: " + e.getMessage());
        }
        
    }
    
    /**
     * Crea un ArrayList existente en el archivo de la dirección saveDir
     * @param saveDir directorio del que se va a extraer el guardadoz
     * @return Arraylist guardado en el archivo con dirección saveDir
     */
    public static ArrayList downloadSave(String saveDir){
        ArrayList i = new ArrayList();
        try{
            FileInputStream iStream = new FileInputStream(saveDir);
            ObjectInputStream oiStream = new ObjectInputStream(iStream);
            i = (ArrayList) oiStream.readObject();
            iStream.close();
        }
        catch (IOException ioe){
            System.out.println("Error de I/O: " + ioe.getMessage());
        }
        catch (Exception e){
            System.out.println("Error: " + e.getMessage());
        }
        return i;
    }

    public static ArrayList<Producto> getCarritoActual() {
        return carritoActual;
    }

    public static void setCarritoActual(ArrayList<Producto> carrito) {
        DataManager.carritoActual = carrito;
    }

    public static ArrayList<Cliente> getClientes() {
        return clientes;
    }

    public static void setClientes(ArrayList<Cliente> clientes) {
        DataManager.clientes = clientes;
    }

    public static ArrayList<Producto> getProductos() {
        return productos;
    }

    public static void setProductos(ArrayList<Producto> productos) {
        DataManager.productos = productos;
    }

    public static ArrayList<Pedido> getPedidos() {
        return pedidos;
    }

    public static void setPedidos(ArrayList<Pedido> pedidos) {
        DataManager.pedidos = pedidos;
    }
}
