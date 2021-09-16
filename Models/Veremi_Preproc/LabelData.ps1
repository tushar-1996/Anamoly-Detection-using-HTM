 foreach($file_location in Get-ChildItem -Name -Include @('ConstPos_0709','DataReplaySybil_0709')){
 $hash = @{}
 cd $file_location   
    foreach($file_location in Get-ChildItem -Name -Exclude *.ps1){
    echo $file_location

    cd $file_location
    $tile_size = 1

    foreach($GT_file in Get-ChildItem -Filter traceGroundTruth*){
        $json = Get-Content -Path $GT_file.name | ConvertFrom-JSON
        $pos_x = foreach($node in $json){$node.pos[0]}
        $pos_y = foreach($node in $json){$node.pos[1]}
        $x_maxmin = $pos_x | measure -Minimum
        $y_maxmin = $pos_x | measure -Minimum
    }

    foreach($file in Get-ChildItem -Filter traceJSON*){
        $attr = $file.name.split("-")
        if("A17", "A1" -Contains $attr[3]){
            $hash[$attr[1]] = 1
         }
         else{
            $hash[$attr[1]] = 0
         }
     }
    cd ..
    }
$x_maxmin.Minimum, $y_maxmin.Minimum | Out-File "minimums.txt"
[PSCustomObject]$hash | Export-CSV -NoTypeInformation -Path "groundtruth.csv"
cd ..
}