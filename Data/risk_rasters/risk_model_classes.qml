<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <pipe>
    <rasterrenderer band="1" type="paletted" opacity="1" nodataColor="" alphaBand="-1">
      <rasterTransparency/>
      <colorPalette>
          <paletteEntry value="0" alpha="255" color="#f2f0eb" label="Background"/>
          <paletteEntry value="1" alpha="255" color="#ffe08a" label="LOW - top 25%"/>
          <paletteEntry value="2" alpha="255" color="#f79a3e" label="MEDIUM - top 10%"/>
          <paletteEntry value="3" alpha="255" color="#c1272d" label="HIGH - top 5%"/>
      </colorPalette>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation colorizeOn="0" saturation="0" grayscaleMode="0"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
