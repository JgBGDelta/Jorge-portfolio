/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JPanel.java to edit this template
 */
package com.JavaComp.interf;

import com.JavaComp.program.DataManager;
import com.JavaComp.program.Producto;
import java.awt.Image;
import javax.swing.ImageIcon;

/**
 *
 * @author Slend
 */
public class DisplayInventario extends javax.swing.JPanel {

    /**
     * Inicializa el display para mostrar todos los datos del producto prod
     * @param prod el producto a representar
     * @param parent el frame que contiene al display
     */
    public DisplayInventario(Producto prod, Inventario parent) {
        initComponents();
        this.prod = prod;
        this.parent = parent;
        
        ratingLabel.setText("Rating: " + String.format("%.1f", prod.getMidRating()));
        
        stockLabel.setText("Stock: " + prod.getStock());
        fechaLabel.setText(prod.getFechaEntrada().toString());
        precioLabel.setText(Double.toString(prod.getPvp()) + "€");
        descripcionArea.setText(prod.getCaracteristicas());
        categoriaLabel.setText("Categoría: " + prod.getCategoria());
        tituloLabel.setText(prod.getTitulo());
        
        ImageIcon imageIcon = new ImageIcon(prod.getImagen()); // load the image to a imageIcon
        Image image = imageIcon.getImage(); // transform it 
        Image newimg = image.getScaledInstance(155, 155,  java.awt.Image.SCALE_SMOOTH); // scale it the smooth way  
        imageIcon = new ImageIcon(newimg);
        imagenLabel.setText("");
        imagenLabel.setIcon(imageIcon);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        modificarBoton = new javax.swing.JButton();
        eliminarBoton = new javax.swing.JButton();
        imagenLabel = new javax.swing.JLabel();
        tituloLabel = new javax.swing.JLabel();
        precioLabel = new javax.swing.JLabel();
        stockLabel = new javax.swing.JLabel();
        fechaLabel = new javax.swing.JLabel();
        descripcionScroll = new javax.swing.JScrollPane();
        descripcionArea = new javax.swing.JTextArea();
        ratingLabel = new javax.swing.JLabel();
        categoriaLabel = new javax.swing.JLabel();

        modificarBoton.setText("Modificar");
        modificarBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                modificarBotonActionPerformed(evt);
            }
        });

        eliminarBoton.setText("Eliminar");
        eliminarBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                eliminarBotonActionPerformed(evt);
            }
        });

        imagenLabel.setText("Imagen");

        tituloLabel.setFont(new java.awt.Font("Segoe UI", 0, 24)); // NOI18N
        tituloLabel.setText("Titulo");

        precioLabel.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        precioLabel.setText("Precio");

        stockLabel.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        stockLabel.setHorizontalAlignment(javax.swing.SwingConstants.TRAILING);
        stockLabel.setText("Stock");

        fechaLabel.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        fechaLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        fechaLabel.setText("Fecha");

        descripcionArea.setEditable(false);
        descripcionArea.setColumns(20);
        descripcionArea.setLineWrap(true);
        descripcionArea.setRows(5);
        descripcionArea.setWrapStyleWord(true);
        descripcionScroll.setViewportView(descripcionArea);

        ratingLabel.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        ratingLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        ratingLabel.setText("Rating");

        categoriaLabel.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        categoriaLabel.setText("Categoria");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(categoriaLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(modificarBoton)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(eliminarBoton))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(imagenLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 155, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(tituloLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(precioLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 83, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(fechaLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 120, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(ratingLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 88, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(stockLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 124, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(descripcionScroll))))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(imagenLabel, javax.swing.GroupLayout.DEFAULT_SIZE, 155, Short.MAX_VALUE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(tituloLabel)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(fechaLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 20, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                .addComponent(precioLabel)
                                .addComponent(stockLabel)
                                .addComponent(ratingLabel)))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(descripcionScroll)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(modificarBoton)
                    .addComponent(eliminarBoton)
                    .addComponent(categoriaLabel))
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void modificarBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_modificarBotonActionPerformed
        ModificarProducto interf = new ModificarProducto(prod);
        interf.setLocationRelativeTo(null);
        interf.setVisible(true);
        parent.setVisible(false);
        parent.dispose();
    }//GEN-LAST:event_modificarBotonActionPerformed

    private void eliminarBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_eliminarBotonActionPerformed
        int opcion = javax.swing.JOptionPane.showConfirmDialog(parent, 
                "¿Está seguro de que desea eliminar el producto \"" + prod.getTitulo() 
                + "\" permanentemente?", "Confirmar eliminación", 
                javax.swing.JOptionPane.OK_CANCEL_OPTION,javax.swing.JOptionPane.QUESTION_MESSAGE);
        switch (opcion){
            case 0:
                DataManager.getProductos().remove(prod);
                DataManager.displayInventario(DataManager.getProductos(), parent.getInventarioPanel(), parent);
                break;
        }
    }//GEN-LAST:event_eliminarBotonActionPerformed

    private Inventario parent;
    private Producto prod;
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel categoriaLabel;
    private javax.swing.JTextArea descripcionArea;
    private javax.swing.JScrollPane descripcionScroll;
    private javax.swing.JButton eliminarBoton;
    private javax.swing.JLabel fechaLabel;
    private javax.swing.JLabel imagenLabel;
    private javax.swing.JButton modificarBoton;
    private javax.swing.JLabel precioLabel;
    private javax.swing.JLabel ratingLabel;
    private javax.swing.JLabel stockLabel;
    private javax.swing.JLabel tituloLabel;
    // End of variables declaration//GEN-END:variables
}