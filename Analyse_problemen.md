# Analyse problemen

Wanneer we in google colab de analyse-files willen aanmaken zien we dat dit niet mogelijk is wegens dat we aan het maximum aantal recources zitten dat google colab ons kan geven.  
We hebben een report1.nsys-rep file kunnen aanmaken maar deze is zonder de "Cuda-HW". Deze kan niet aangemaakt worden wegens te weinig recources.  
We hebben alle commands in onze google colab staan maar deze werken dus niet. De analyse kan dus niet worden uitgevoerd...

Dit is ook getest via de WSL-commandprompt maar hiermee werkt het ook niet.    
Als laatste optie hebben we dit geprobeerd in een Virtual Machine waarin Ubuntu staat geïnstalleerd.  
Hierin werkt het ook niet aangezien deze hier de nvidia-toolbox niet in geïnstalleerd krijgt. (staat wel geïnstallerd maar werkt niet, hij kent nvcc niet)  
