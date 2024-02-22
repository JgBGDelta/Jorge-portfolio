/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package com.JavaComp.interf;

import javax.swing.ImageIcon;
import com.JavaComp.program.*;
import java.awt.Image;
import javax.swing.JOptionPane;

/**
 *
 * @author Slend
 */
public class SignIn extends javax.swing.JFrame {

    /**
     * Creates new form SignIn
     */
    public SignIn() {
        initComponents();
        ImageIcon imageIcon = new ImageIcon("src/main/resources/images/LogoJavaComp.png"); // load the image to a imageIcon
        Image image = imageIcon.getImage(); // transform it 
        Image newimg = image.getScaledInstance(100, 100,  java.awt.Image.SCALE_SMOOTH); // scale it the smooth way  
        imageIcon = new ImageIcon(newimg);
        iconImageLabel.setIcon(imageIcon);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        correo = new javax.swing.JLabel();
        correoField = new javax.swing.JTextField();
        clave = new javax.swing.JLabel();
        claveField = new javax.swing.JPasswordField();
        iconImageLabel = new javax.swing.JLabel();
        cancelBoton = new javax.swing.JButton();
        signInBoton = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("JavaComp");
        setIconImage(new javax.swing.ImageIcon("src/main/resources/images/LogoJavaComp.png").getImage());

        correo.setText("Correo:");

        correoField.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                correoFieldActionPerformed(evt);
            }
        });

        clave.setText("Clave de acceso:");

        claveField.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                claveFieldActionPerformed(evt);
            }
        });

        cancelBoton.setBackground(new java.awt.Color(240, 152, 28));
        cancelBoton.setFont(new java.awt.Font("Segoe UI", 1, 12)); // NOI18N
        cancelBoton.setForeground(new java.awt.Color(0, 0, 0));
        cancelBoton.setText("Cancel");
        cancelBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0), 3));
        cancelBoton.setFocusPainted(false);
        cancelBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cancelBotonActionPerformed(evt);
            }
        });

        signInBoton.setBackground(new java.awt.Color(240, 152, 28));
        signInBoton.setForeground(new java.awt.Color(0, 0, 0));
        signInBoton.setText("Sign In");
        signInBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0), 3));
        signInBoton.setFocusPainted(false);
        signInBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                signInBotonActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap(28, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(correo)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(correoField, javax.swing.GroupLayout.PREFERRED_SIZE, 234, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(clave)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(cancelBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 70, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(signInBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 70, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addComponent(claveField, javax.swing.GroupLayout.PREFERRED_SIZE, 234, javax.swing.GroupLayout.PREFERRED_SIZE))))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(iconImageLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(33, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(correo, javax.swing.GroupLayout.PREFERRED_SIZE, 22, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(correoField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(claveField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(clave))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cancelBoton)
                    .addComponent(signInBoton))
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(layout.createSequentialGroup()
                .addContainerGap(26, Short.MAX_VALUE)
                .addComponent(iconImageLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(26, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    //Comprueba que exista la cuenta con la que se quiera entrar: Si es de admin opciones de admin y si es de cliente opciones de cliente
    private void signInBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_signInBotonActionPerformed
        // TODO add your handling code here:
        String corr = correoField.getText();
        String cont = claveField.getText();
        
        if (corr.equals("admin@javacomp.com") && cont.equals("admin")){
            InterfAdmin interf = new InterfAdmin();
            interf.setLocationRelativeTo(null);
            interf.setVisible(true);
            this.setVisible(false);
            this.dispose();
        }
        else{
            boolean logged = Cliente.LoggearCliente(corr, cont);
            if (logged){
                InterfCliente interfClient = new InterfCliente();
                interfClient.setLocationRelativeTo(null);
                interfClient.setVisible(true);
                this.setVisible(false);
            }
            else JOptionPane.showMessageDialog(this, "Correo o clave incorrectos", "Error", JOptionPane.WARNING_MESSAGE);
        }
    }//GEN-LAST:event_signInBotonActionPerformed
    // Regresa al menu principal
    private void cancelBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cancelBotonActionPerformed
        prevFrame.setLocationRelativeTo(null);
        prevFrame.setVisible(true);
        this.setVisible(false);
    }//GEN-LAST:event_cancelBotonActionPerformed

    private void claveFieldActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_claveFieldActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_claveFieldActionPerformed

    private void correoFieldActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_correoFieldActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_correoFieldActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(SignIn.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(SignIn.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(SignIn.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(SignIn.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new SignIn().setVisible(true);
            }
        });
    }
    public javax.swing.JFrame prevFrame;
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton cancelBoton;
    private javax.swing.JLabel clave;
    private javax.swing.JPasswordField claveField;
    private javax.swing.JLabel correo;
    private javax.swing.JTextField correoField;
    private javax.swing.JLabel iconImageLabel;
    private javax.swing.JButton signInBoton;
    // End of variables declaration//GEN-END:variables
}