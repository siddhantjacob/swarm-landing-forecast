<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer band="1" type="paletted" opacity="1" nodataColor="" alphaBand="-1">
      <rasterTransparency/>
      <colorPalette>
          <paletteEntry value="0" alpha="0" color="#f2f0eb" label="Outside buffer"/>
          <paletteEntry value="1" alpha="255" color="#9aa5b1" label="Within 30 km of known band activity"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0" saturation="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
