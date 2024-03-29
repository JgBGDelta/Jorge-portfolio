/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JPanel.java to edit this template
 */
package com.JavaComp.interf;

import com.JavaComp.program.Producto;
import java.awt.*;
import java.io.File;
import javax.swing.ImageIcon;

/**
 *
 * @author Slend
 */
public class DisplayProducto extends javax.swing.JPanel {

    /**
     * Creates new form DisplayProducto
     */
    public DisplayProducto() {
        initComponents();
    }
    
    public void setParameters(){
        
        Precio.setText((Double.toString(prod.getPvp())) + "€");
        Precio.setVisible(true);
        imagenLabel.setText("");
        imagenLabel.setVisible(true);
        tituloLabel.setText(prod.getTitulo());
        tituloLabel.setVisible(true);
        caracteristicasArea.setText(prod.getCaracteristicas());
        
        System.out.print(new File(prod.getImagen()).exists() + "\n");
        System.out.print(prod.getImagen() + "\n");
        ImageIcon imageIcon = new ImageIcon(prod.getImagen()); // load the image to a imageIcon
        Image image = imageIcon.getImage(); // transform it 
        Image newimg = image.getScaledInstance(163, 163,  java.awt.Image.SCALE_SMOOTH); // scale it the smooth way  
        imageIcon = new ImageIcon(newimg);
        imagenLabel.setIcon(imageIcon);
        imagenLabel.setText("");
        
        repaint();
        revalidate();
    }
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        imagenLabel = new javax.swing.JLabel();
        Precio = new javax.swing.JLabel();
        tituloLabel = new javax.swing.JLabel();
        caracteristicasScroll = new javax.swing.JScrollPane();
        caracteristicasArea = new javax.swing.JTextArea();
        verBoton = new javax.swing.JButton();

        setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(102, 102, 102), 2));

        imagenLabel.setText("Imagen");
        imagenLabel.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 2));

        Precio.setHorizontalAlignment(javax.swing.SwingConstants.TRAILING);
        Precio.setText("Precio");

        tituloLabel.setFont(new java.awt.Font("Segoe UI", 1, 14)); // NOI18N
        tituloLabel.setText("Titulo");

        caracteristicasScroll.setBorder(null);
        caracteristicasScroll.setViewportBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 2));

        caracteristicasArea.setEditable(false);
        caracteristicasArea.setBackground(new java.awt.Color(255, 220, 181));
        caracteristicasArea.setColumns(20);
        caracteristicasArea.setLineWrap(true);
        caracteristicasArea.setRows(5);
        caracteristicasArea.setWrapStyleWord(true);
        caracteristicasScroll.setViewportView(caracteristicasArea);

        verBoton.setBackground(new java.awt.Color(255, 191, 29));
        verBoton.setForeground(new java.awt.Color(51, 51, 51));
        verBoton.setText("Ver");
        verBoton.setFocusPainted(false);
        verBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                verBotonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(imagenLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 163, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(caracteristicasScroll))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(tituloLabel, javax.swing.GroupLayout.DEFAULT_SIZE, 347, Short.MAX_VALUE)
                        .addGap(117, 117, 117)
                        .addComponent(Precio, javax.swing.GroupLayout.PREFERRED_SIZE, 74, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(verBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 60, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(caracteristicasScroll, javax.swing.GroupLayout.DEFAULT_SIZE, 163, Short.MAX_VALUE)
                    .addComponent(imagenLabel, javax.swing.GroupLayout.DEFAULT_SIZE, 163, Short.MAX_VALUE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(Precio)
                    .addComponent(verBoton)
                    .addComponent(tituloLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addContainerGap())
        );
    }// </editor-fold>//GEN-END:initComponents

    private void verBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_verBotonActionPerformed
        // TODO add your handling code here:
        InterfProducto interf = new InterfProducto();
        interf.setLocationRelativeTo(null);
        interf.setProd(prod);
        interf.setInterfaz();
        interf.setVisible(true);
    }//GEN-LAST:event_verBotonActionPerformed
    
    public void setProd(Producto prod) {
        this.prod = prod;
    }
    private Producto prod;
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel Precio;
    private javax.swing.JTextArea caracteristicasArea;
    private javax.swing.JScrollPane caracteristicasScroll;
    private javax.swing.JLabel imagenLabel;
    private javax.swing.JLabel tituloLabel;
    private javax.swing.JButton verBoton;
    // End of variables declaration//GEN-END:variables
}
