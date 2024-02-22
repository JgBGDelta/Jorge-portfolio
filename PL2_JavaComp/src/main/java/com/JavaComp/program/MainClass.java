/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package com.JavaComp.program;

import com.JavaComp.interf.*;

/**
 *
 * @author Slend
 */

public class MainClass {
    
    public static void main(String[] args) {
        DataManager.setClientes(DataManager.downloadSave("src/main/resources/saves/saveClientes.dat"));
        DataManager.setProductos(DataManager.downloadSave("src/main/resources/saves/saveProductos.dat"));
        DataManager.setPedidos(DataManager.downloadSave("src/main/resources/saves/savePedidos.dat"));
        for (int i=0; i<DataManager.getClientes().size();i++){
            System.out.print(DataManager.getClientes().get(i).getCorreo() + " ; " + DataManager.getClientes().get(i).getClave() + "\n");
        }
        for (int i=0; i<DataManager.getProductos().size();i++){
            System.out.print(DataManager.getProductos().get(i).getTitulo() + " . " + DataManager.getProductos().get(i).getPvp() + "\n");
        }

        MainMenu main = new MainMenu();
        main.setLocationRelativeTo(null);
        main.setVisible(true);
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            public void run() {
                DataManager.uploadSave(DataManager.getClientes(), "src/main/resources/saves/saveClientes.dat");
                DataManager.uploadSave(DataManager.getProductos(), "src/main/resources/saves/saveProductos.dat");
                DataManager.uploadSave(DataManager.getPedidos(), "src/main/resources/saves/savePedidos.dat");
                for (int i=0; i<DataManager.getClientes().size();i++){
                    System.out.print(DataManager.getClientes().get(i).getCorreo() + " : " + DataManager.getClientes().get(i).getClave() +"\n");
                }
                System.out.println(DataManager.getPedidos().size());
            }
        }));
    }
}